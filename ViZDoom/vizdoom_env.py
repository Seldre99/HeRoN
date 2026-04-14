"""
VizDoom Environment Wrapper for HERON with Oracle Mode.

Supports scenarios: Basic, Deadly Corridor, Defend The Center

Oracle Mode provides numerical feature vectors instead of pixels,
transforming the problem from visual computing to decision-making.

"""

import numpy as np
import cv2
from vizdoom import DoomGame, Mode, ScreenFormat, ScreenResolution, GameVariable
from collections import deque


# Common enemy names in VizDoom
ENEMY_NAMES = [
    'Marine', 'Chainsaw', 'Demon', 'Imp', 'Zombieman', 'ShotgunGuy',
    'ChaingunGuy', 'Cacodemon', 'Baron', 'Knight', 'Revenant',
    'Arachnotron', 'Mancubus', 'Archvile', 'Cyberdemon', 'Mastermind'
]

# Action name aliases for normalization
ACTION_ALIASES = {
    'SHOOT': 'ATTACK',
    'FIRE': 'ATTACK',
    'LEFT': 'MOVE_LEFT',
    'RIGHT': 'MOVE_RIGHT',
    'FORWARD': 'MOVE_FORWARD',
    'BACKWARD': 'MOVE_BACKWARD',
    'BACK': 'MOVE_BACKWARD',
    'STRAFE_LEFT': 'MOVE_LEFT',
    'STRAFE_RIGHT': 'MOVE_RIGHT',
    'ROTATE_LEFT': 'TURN_LEFT',
    'ROTATE_RIGHT': 'TURN_RIGHT'
}


class VizDoomEnv:
    """
    VizDoom environment wrapper compatible with HERON architecture.

    Oracle Mode (use_oracle=True):
        State = numerical vector (6-10 values) with ground truth
        Uses simple MLP, much faster training

    Visual Mode (use_oracle=False):
        State = stacked frames
        Requires CNN, more realistic but harder
    """

    SCENARIOS = {
        'basic': {
            'config': 'basic.cfg',
            'wad': 'basic.wad',
            'actions': ['MOVE_LEFT', 'MOVE_RIGHT', 'ATTACK'],
            'objective': 'Shoot the monster in front of you',
            'description': 'A simple room with one monster ahead'
        },
        'deadly_corridor': {
            'config': 'deadly_corridor.cfg',
            'wad': 'deadly_corridor.wad',
            'actions': ['MOVE_LEFT', 'MOVE_RIGHT', 'ATTACK', 'MOVE_FORWARD',
                       'MOVE_BACKWARD', 'TURN_LEFT', 'TURN_RIGHT'],
            'objective': 'Navigate through the corridor and reach the green vest',
            'description': 'A corridor with enemies on both sides'
        },
        'defend_the_center': {
            'config': 'defend_the_center.cfg',
            'wad': 'defend_the_center.wad',
            'actions': ['TURN_LEFT', 'TURN_RIGHT', 'ATTACK'],
            'objective': 'Survive by killing approaching monsters',
            'description': 'Circular arena with monsters approaching'
        }
    }

    def __init__(self, scenario='deadly_corridor', frame_stack=4, frame_size=(84, 84),
                 visible=False, config_path=None, use_oracle=True):
        """
        Initialize the VizDoom environment.

        Args:
            scenario: Task name ('basic', 'deadly_corridor', 'defend_the_center')
            frame_stack: Number of frames to stack (ignored if use_oracle=True)
            frame_size: Preprocessed frame dimensions (ignored if use_oracle=True)
            visible: Show game window
            config_path: Custom path for configuration files
            use_oracle: Use numerical feature vectors instead of pixels
        """
        self.scenario_name = scenario
        self.scenario = self.SCENARIOS[scenario]
        self.frame_stack = frame_stack
        self.frame_size = frame_size
        self.use_oracle = use_oracle

        self.game = DoomGame()

        config_dir = config_path or "doom_files"
        self.game.load_config(f"{config_dir}/{self.scenario['config']}")
        self.game.set_doom_scenario_path(f"{config_dir}/{self.scenario['wad']}")

        self.game.set_window_visible(visible)
        self.game.set_mode(Mode.PLAYER)
        self.game.set_screen_format(ScreenFormat.RGB24)
        self.game.set_screen_resolution(ScreenResolution.RES_640X480)
        self.game.set_labels_buffer_enabled(True)
        self.sleep_time = 0.028 if not visible else 0.001

        self.action_names = self.scenario['actions']
        self.action_size = len(self.action_names)
        self.possible_actions = np.eye(self.action_size, dtype=int).tolist()

        if not use_oracle:
            self.frames = deque(maxlen=frame_stack)

        self._reset_tracking_vars()

        if use_oracle:
            self.oracle_feature_size = 10
            self.state_shape = (self.oracle_feature_size,)
        else:
            self.state_shape = (frame_stack, frame_size[0], frame_size[1])

    def _reset_tracking_vars(self):
        """Reset all tracking variables."""
        self.last_health = 100
        self.last_ammo = 0
        self.last_kill_count = 0
        self.episode_reward = 0
        self.episode_length = 0
        self.current_step = 0
        self.last_position_x = 0
        self.max_progress = 0
        self.consecutive_hits = 0
        self.time_without_progress = 0
        self.last_armor = 0
        self.vest_acquired = False
        self.damage_history = deque(maxlen=5)
        self.last_helper_action = None
        self.last_helper_step = -100

    def init(self):
        """Initialize the VizDoom game."""
        self.game.init()

    def close(self):
        """Close the VizDoom game."""
        self.game.close()

    def get_oracle_state(self):
        """
        Extract numerical state vector from Labels Buffer (ground truth).

        Returns:
            np.array of shape (oracle_feature_size,) with normalized values [0, 1]

        Features (10 values):
            [0] Health normalized [0-1]
            [1] Ammo normalized [0-1]
            [2] Armor normalized [0-1]
            [3] Enemies visible LEFT (normalized)
            [4] Enemies visible CENTER (normalized)
            [5] Enemies visible RIGHT (normalized)
            [6] Enemies in CROSSHAIR (normalized)
            [7] Distance to closest enemy (normalized, 1=far, 0=close)
            [8] Kill count normalized
            [9] Damage taken recently (binary)
        """
        health = self._get_health() / 100.0
        ammo = min(50.0, self._get_ammo()) / 50.0
        armor = self._get_armor() / 100.0
        kill_count = min(20.0, self._get_kill_count()) / 20.0
        damage_taken = 1.0 if self._took_damage_recently() else 0.0

        state = self.game.get_state()
        enemies_left = 0
        enemies_center = 0
        enemies_right = 0
        enemies_crosshair = 0
        closest_dist = 1.0

        if state and state.labels:
            screen_w = self.game.get_screen_width()
            screen_center_x = screen_w / 2
            crosshair_tolerance = int(screen_w * 0.08)

            for label in state.labels:
                obj_name = label.object_name

                if 'Player' in obj_name or 'Doom' in obj_name:
                    continue
                if 'Puff' in obj_name or 'Bullet' in obj_name or 'Blood' in obj_name:
                    continue

                is_enemy = any(enemy in obj_name for enemy in ENEMY_NAMES)

                if is_enemy:
                    x_pos = label.x + label.width / 2

                    if x_pos < screen_w * 0.35:
                        enemies_left += 1
                    elif x_pos > screen_w * 0.65:
                        enemies_right += 1
                    else:
                        enemies_center += 1
                        if abs(x_pos - screen_center_x) <= crosshair_tolerance:
                            enemies_crosshair += 1

                    dist = 1.0 - min(1.0, label.width / 150.0)
                    if dist < closest_dist:
                        closest_dist = dist

        enemies_left = min(1.0, enemies_left / 5.0)
        enemies_center = min(1.0, enemies_center / 5.0)
        enemies_right = min(1.0, enemies_right / 5.0)
        enemies_crosshair = min(1.0, enemies_crosshair / 3.0)

        return np.array([
            health, ammo, armor,
            enemies_left, enemies_center, enemies_right, enemies_crosshair,
            closest_dist, kill_count, damage_taken
        ], dtype=np.float32)

    def preprocess_frame(self, frame):
        """
        Preprocess a single frame: resize, grayscale, normalize.
        Used only in Visual Mode.
        """
        if frame is None:
            return np.zeros(self.frame_size, dtype=np.float32)

        if len(frame.shape) == 3:
            if frame.shape[0] in [3, 4]:
                frame = np.transpose(frame, (1, 2, 0))
            elif frame.shape[2] > 4:
                frame = np.transpose(frame, (1, 2, 0))

            if not frame.flags['C_CONTIGUOUS']:
                frame = np.ascontiguousarray(frame, dtype=frame.dtype)

            if frame.shape[2] == 3:
                gray = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
            elif frame.shape[2] == 4:
                gray = cv2.cvtColor(frame, cv2.COLOR_RGBA2GRAY)
            else:
                gray = frame[:, :, 0]
        elif len(frame.shape) == 2:
            gray = frame
        else:
            gray = np.zeros(self.frame_size, dtype=np.float32)

        if self.scenario_name == 'deadly_corridor':
            h, w = gray.shape
            crop_top = int(h * 0.1)
            crop_bottom = int(h * 0.9)
            gray = gray[crop_top:crop_bottom, :]

        resized = cv2.resize(gray, self.frame_size, interpolation=cv2.INTER_AREA)
        normalized = resized.astype(np.float32) / 255.0

        return normalized

    def stack_frames(self, frame, is_new_episode=False):
        """Add a frame to the stack and return the complete stack."""
        if is_new_episode:
            self.frames.clear()
            for _ in range(self.frame_stack):
                self.frames.append(frame)
        else:
            self.frames.append(frame)

        return np.array(self.frames, dtype=np.float32)

    def reset(self):
        """
        Reset the environment for a new episode.

        Returns:
            Initial state (Oracle vector or frame stack)
        """
        self.game.new_episode()
        self._reset_tracking_vars()

        self.last_health = self._get_health()
        self.last_ammo = self._get_ammo()
        self.last_kill_count = self._get_kill_count()

        try:
            self.last_position_x = self.game.get_game_variable(GameVariable.POSITION_X)
            self.last_armor = self.game.get_game_variable(GameVariable.ARMOR)
        except:
            pass

        if self.use_oracle:
            return self.get_oracle_state()
        else:
            frame = self.game.get_state().screen_buffer
            if frame is not None:
                frame = frame.transpose(1, 2, 0)
            processed = self.preprocess_frame(frame)
            return self.stack_frames(processed, is_new_episode=True)

    def step(self, action):
        """
        Execute an action in the environment.

        Args:
            action: Action index to execute

        Returns:
            Tuple (next_state, reward, done, info)
        """
        skip_tics = 12

        reward = self.game.make_action(self.possible_actions[action], skip_tics)
        done = self.game.is_episode_finished()
        shaped_reward = self._shape_reward(reward, done)

        self.episode_reward += shaped_reward
        self.episode_length += 1
        self.current_step += 1

        if done:
            if self.use_oracle:
                next_state = np.zeros(self.state_shape, dtype=np.float32)
            else:
                next_state = np.zeros((self.frame_stack, self.frame_size[0],
                                       self.frame_size[1]), dtype=np.float32)
            info = self._get_episode_info()
        else:
            if self.use_oracle:
                next_state = self.get_oracle_state()
            else:
                frame = self.game.get_state().screen_buffer
                if frame is not None:
                    frame = frame.transpose(1, 2, 0)
                processed = self.preprocess_frame(frame)
                next_state = self.stack_frames(processed)
            info = self._get_step_info()

        return next_state, shaped_reward, done, info

    def _shape_reward(self, base_reward, done):
        """Apply scenario-specific reward shaping."""
        shaped = base_reward

        current_health = self._get_health()
        health_delta = current_health - self.last_health

        if health_delta < 0:
            self.damage_history.append(True)
        else:
            self.damage_history.append(False)

        current_kills = self._get_kill_count()
        kill_delta = current_kills - self.last_kill_count
        if kill_delta > 0:
            shaped += kill_delta * 20
            self.consecutive_hits += 1
            if self.consecutive_hits >= 3:
                shaped += 5
        else:
            self.consecutive_hits = 0
        self.last_kill_count = current_kills
        self.last_health = current_health

        if self.scenario_name == 'deadly_corridor':
            shaped = self._shape_reward_deadly_corridor(shaped, done, health_delta)
        elif self.scenario_name == 'basic':
            shaped = self._shape_reward_basic(shaped, done, health_delta)
        elif self.scenario_name == 'defend_the_center':
            shaped = self._shape_reward_defend(shaped, done, health_delta)
        else:
            if health_delta < 0:
                shaped += health_delta * 0.1
            if done and current_health <= 0:
                shaped -= 50

        return shaped

    def _shape_reward_deadly_corridor(self, shaped, done, health_delta):
        """Reward shaping for Deadly Corridor scenario."""
        current_health = self._get_health()
        current_armor = self._get_armor()

        armor_delta = current_armor - self.last_armor
        if armor_delta > 0 and not self.vest_acquired:
            shaped += 200
            self.vest_acquired = True
        self.last_armor = current_armor

        try:
            current_x = self.game.get_game_variable(GameVariable.POSITION_X)
            progress = current_x - self.last_position_x

            if progress > 0:
                shaped += progress * 0.3
                self.time_without_progress = 0

                if current_x > self.max_progress:
                    shaped += (current_x - self.max_progress) * 0.2
                    self.max_progress = current_x

                    if int(current_x / 300) > int(self.last_position_x / 300):
                        shaped += 30
            elif progress < -10:
                shaped -= 2
                self.time_without_progress += 2
            else:
                self.time_without_progress += 1

            if self.time_without_progress > 40:
                shaped -= (self.time_without_progress - 40) * 0.1

            self.last_position_x = current_x
        except:
            pass

        if health_delta < 0:
            shaped += health_delta * 0.15

        shaped += 0.05

        if done and current_health <= 0:
            if self.vest_acquired:
                shaped -= 10
            else:
                progress_ratio = min(1.0, self.max_progress / 1000)
                death_penalty = 50 * (1 - progress_ratio * 0.7)
                shaped -= death_penalty

        if done and current_health > 0:
            shaped += 300
            shaped += current_health * 1.0
            if self.episode_length < 300:
                shaped += 50

        return shaped

    def _shape_reward_basic(self, shaped, done, health_delta):
        """Reward shaping for Basic scenario."""
        current_health = self._get_health()

        if health_delta < 0:
            shaped += health_delta * 0.05

        if done and current_health <= 0:
            shaped -= 30

        if done and current_health > 0:
            speed_bonus = max(0, 50 - self.episode_length * 0.2)
            shaped += speed_bonus

        return shaped

    def _shape_reward_defend(self, shaped, done, health_delta):
        """Simplified reward shaping for Defend the Center."""
        current_health = self._get_health()
        current_kills = self._get_kill_count()
        kill_delta = current_kills - self.last_kill_count

        if kill_delta > 0:
            self.last_kill_count = current_kills
            shaped += 1.0 * kill_delta

        if done and current_health <= 0:
            shaped -= 1.0

        return shaped

    def _get_health(self):
        """Get current player health."""
        try:
            return self.game.get_game_variable(GameVariable.HEALTH)
        except:
            return 100

    def _get_ammo(self):
        """Get current ammo count."""
        try:
            return self.game.get_game_variable(GameVariable.AMMO2)
        except:
            return 0

    def _get_armor(self):
        """Get current armor value."""
        try:
            return self.game.get_game_variable(GameVariable.ARMOR)
        except:
            return 0

    def _get_kill_count(self):
        """Get current kill count."""
        try:
            return self.game.get_game_variable(GameVariable.KILLCOUNT)
        except:
            return 0

    def _get_step_info(self):
        """Get current step information."""
        return {
            'health': self._get_health(),
            'ammo': self._get_ammo(),
            'armor': self._get_armor(),
            'kills': self._get_kill_count(),
            'episode_length': self.episode_length
        }

    def _get_episode_info(self):
        """Get episode completion information."""
        health = self._get_health()
        victory = health > 0

        return {
            'health': health,
            'ammo': self._get_ammo(),
            'armor': self._get_armor(),
            'kills': self._get_kill_count(),
            'episode_length': self.episode_length,
            'episode_reward': self.episode_reward,
            'victory': victory
        }

    def get_valid_actions(self):
        """Get valid actions for current state. Masks ATTACK when ammo=0."""
        valid = list(range(self.action_size))

        ammo = self._get_ammo()
        if ammo <= 0 and 'ATTACK' in self.action_names:
            attack_idx = self.action_names.index('ATTACK')
            if attack_idx in valid:
                valid.remove(attack_idx)

        if len(valid) == 0:
            valid = list(range(self.action_size))

        return valid

    def describe_game_state(self):
        """
        Generate textual description of current game state for Helper LLM.
        Uses Labels Buffer for accurate enemy detection.
        """
        state = self.game.get_state()
        if state is None:
            return "Game state unavailable."

        health = self._get_health()
        ammo = self._get_ammo()
        armor = self._get_armor()
        kills = self._get_kill_count()
        health_pct = health / 100.0

        screen_width = self.game.get_screen_width()
        screen_height = self.game.get_screen_height()
        screen_center_x = screen_width // 2
        crosshair_tolerance = int(screen_width * 0.08)

        enemy_center = False
        enemy_left = False
        enemy_right = False
        enemies_visible = 0
        enemies_in_crosshair = 0
        enemy_positions = []
        threat_level = "LOW"

        if state.labels is not None:
            for label in state.labels:
                obj_name = label.object_name

                if 'Player' in obj_name or 'Doom' in obj_name:
                    continue
                if 'Puff' in obj_name or 'Bullet' in obj_name or 'Blood' in obj_name:
                    continue

                is_enemy = any(enemy in obj_name for enemy in ENEMY_NAMES)

                if is_enemy:
                    enemies_visible += 1
                    x_pos = label.x

                    if x_pos < screen_width * 0.35:
                        zone = "LEFT"
                        enemy_left = True
                    elif x_pos > screen_width * 0.65:
                        zone = "RIGHT"
                        enemy_right = True
                    else:
                        zone = "CENTER"
                        enemy_center = True

                        if abs(x_pos - screen_center_x) <= crosshair_tolerance:
                            enemies_in_crosshair += 1

                    enemy_positions.append((obj_name, x_pos, zone))

        if enemies_in_crosshair >= 1:
            threat_level = "CRITICAL" if enemies_visible >= 2 else "HIGH"
        elif enemy_center:
            threat_level = "HIGH"
        elif enemies_visible >= 2:
            threat_level = "MODERATE"
        elif enemies_visible >= 1:
            threat_level = "LOW"
        else:
            threat_level = "CLEAR"

        if health_pct > 0.7:
            health_status = "good"
        elif health_pct > 0.3:
            health_status = "moderate"
        else:
            health_status = "critical"

        if ammo > 20:
            ammo_status = "plenty"
        elif ammo > 5:
            ammo_status = "moderate"
        elif ammo > 0:
            ammo_status = "low"
        else:
            ammo_status = "empty"

        took_damage_recently = sum(self.damage_history) >= 2

        enemy_detail = ""
        if enemy_positions:
            enemy_detail = "\n  Detected: " + ", ".join(
                [f"{name}@{zone}" for name, _, zone in enemy_positions[:3]])

        description = f"""{self.scenario_name.replace('_', ' ').title()} - Step {self.current_step}

PLAYER STATUS:
- Health: {health:.0f}% ({health_status})
- Ammo: {ammo} rounds ({ammo_status})
- Armor: {armor}
- Kills this episode: {kills}

ENEMY DETECTION (Oracle - Ground Truth):
- Enemies in CROSSHAIR: {enemies_in_crosshair}
- Enemy in CENTER zone: {'YES' if enemy_center else 'NO'}
- Enemies on LEFT: {'YES' if enemy_left else 'NO'}
- Enemies on RIGHT: {'YES' if enemy_right else 'NO'}
- Total enemies visible: {enemies_visible}{enemy_detail}
- Threat Level: {threat_level}

TACTICAL SITUATION:
- Can hit target now: {'YES' if (enemies_in_crosshair > 0 and ammo > 0) else ('NEED TO AIM' if enemies_visible > 0 else 'NO TARGET')}
- Under fire: {'YES' if took_damage_recently else 'NO'}

OBJECTIVE: {self.scenario['objective']}
AVAILABLE ACTIONS: {', '.join(self.action_names)}"""

        tactical_hints = []

        if enemies_in_crosshair > 0 and ammo > 0:
            tactical_hints.append("ENEMY IN CROSSHAIR - ATTACK NOW")
        elif enemy_center and ammo > 0:
            tactical_hints.append("Enemy in center - fine-tune aim and ATTACK")

        if enemy_left and not enemy_center and not enemies_in_crosshair:
            tactical_hints.append("Enemy on LEFT - TURN_LEFT to aim")
        if enemy_right and not enemy_center and not enemies_in_crosshair:
            tactical_hints.append("Enemy on RIGHT - TURN_RIGHT to aim")

        if enemies_visible == 0:
            tactical_hints.append("No enemies visible - TURN to scan for targets")

        if ammo == 0:
            tactical_hints.append("NO AMMO - Cannot attack")
        elif ammo <= 5:
            tactical_hints.append("LOW AMMO - Make every shot count")

        if health_status == "critical":
            tactical_hints.append("CRITICAL HEALTH - Kill enemies fast")

        if took_damage_recently:
            if enemies_in_crosshair > 0:
                tactical_hints.append("TAKING DAMAGE - ATTACK to eliminate threat")
            else:
                tactical_hints.append("TAKING DAMAGE - Turn to find attacker")

        if tactical_hints:
            description += "\n\nTACTICAL HINTS:\n" + "\n".join(tactical_hints)

        if self.scenario_name == 'deadly_corridor':
            progress_pct = min(100, int((self.max_progress / 1000) * 100))
            description += f"\n\nMISSION: Reach green vest (Progress: {progress_pct}%)"
            if enemies_visible > 0:
                description += "\nClear path: eliminate enemies, then MOVE_FORWARD"
            else:
                description += "\nPath clear: MOVE_FORWARD to advance"

        return description

    def _took_damage_recently(self):
        """Check if damage was taken in recent steps."""
        return sum(self.damage_history) >= 2

    def set_last_helper_action(self, action_name):
        """Record the last Helper-suggested action."""
        self.last_helper_action = action_name
        self.last_helper_step = self.current_step

    def get_action_from_name(self, action_name):
        """
        Convert action name to index.

        Args:
            action_name: Action name string

        Returns:
            Action index or None if not found
        """
        action_name = action_name.upper().strip()
        action_name = ACTION_ALIASES.get(action_name, action_name)

        if action_name in self.action_names:
            return self.action_names.index(action_name)

        return None


def create_vizdoom_env(scenario='deadly_corridor', use_oracle=True, **kwargs):
    """
    Factory function to create a VizDoomEnv.

    Args:
        scenario: Task name
        use_oracle: Use Oracle Mode (numerical vectors)
        **kwargs: Additional arguments for VizDoomEnv

    Returns:
        Initialized VizDoomEnv instance
    """
    env = VizDoomEnv(scenario=scenario, use_oracle=use_oracle, **kwargs)
    env.init()
    return env
