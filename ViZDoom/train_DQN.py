#!/usr/bin/env python3
"""
DQN Training with HERON Strategies and Oracle Mode.

Implements NPC:
- Level 1: DQN Agent (action execution)
- Level 2: Helper LLM (tactical suggestions)
- Level 3: Reviewer LLM (plan correction with memory)

"""

import argparse
import json
import os
import re
import sys
from collections import deque
from datetime import datetime

import numpy as np
import requests

# Configure TensorFlow memory management before imports
os.environ['TF_FORCE_GPU_ALLOW_GROWTH'] = 'true'
os.environ['TF_GPU_ALLOCATOR'] = 'cuda_malloc_async'

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from vizdoom_env import VizDoomEnv, create_vizdoom_env
from vizdoom_agent import DQNAgent, DQNCnnAgent
from heron_strategies import HERONTrainer, parse_strategy
from reviewer_llm import ReviewerLLM


# Configuration from environment variables with defaults
LM_STUDIO_URL = os.getenv('LM_STUDIO_URL', 'http://localhost:1234/v1/chat/completions')
HELPER_MODEL = os.getenv('HELPER_MODEL', 'qwen2.5-vl-7b')
REVIEWER_MODEL_PATH = os.getenv('REVIEWER_MODEL_PATH', './models/reviewer.gguf')
REVIEWER_QUANTIZATION = os.getenv('REVIEWER_QUANTIZATION', '8bit')
REVIEWER_HISTORY_SIZE = int(os.getenv('REVIEWER_HISTORY_SIZE', '5'))
MAX_HELPER_CALLS_PER_EPISODE = 20
PLAN_SIZE = 5


def check_lmstudio_connection():
    """Verify connection to LM Studio."""
    try:
        url = LM_STUDIO_URL.replace("/chat/completions", "/models")
        response = requests.get(url, timeout=2)
        return response.status_code == 200
    except Exception:
        return False


def get_oracle_description(env, state):
    """Get textual state description from environment."""
    return env.describe_game_state()


def create_tactical_prompt(game_state_desc, available_actions, scenario_name, plan_size=5):
    """Create tactical prompt for Helper LLM."""
    actions_str = ', '.join(available_actions)
    return f"""You are a tactical AI for Doom.
CURRENT STATE:
{game_state_desc}
AVAILABLE ACTIONS: {actions_str}
Suggest {plan_size} actions as a JSON array.
OUTPUT FORMAT: ["ACTION1", "ACTION2", ...]
"""


def call_helper_llm(game_state_desc, available_actions, scenario_name, plan_size=5):
    """Call Helper LLM for action suggestions."""
    prompt = create_tactical_prompt(game_state_desc, available_actions, scenario_name, plan_size)
    payload = {
        "model": HELPER_MODEL,
        "messages": [
            {"role": "system", "content": "You are a tactical Doom AI. Output JSON only."},
            {"role": "user", "content": prompt}
        ],
        "temperature": 0.0,
        "max_tokens": 100,
        "stream": False
    }

    try:
        response = requests.post(LM_STUDIO_URL, json=payload, timeout=10)
        if response.status_code == 200:
            content = response.json()['choices'][0]['message']['content']
            return parse_action_plan(content, available_actions)
    except Exception:
        pass
    return []


def parse_action_plan(response, valid_actions):
    """Parse action plan from LLM response."""
    try:
        clean = re.sub(r"```json|```", "", response).strip()
        match = re.search(r'\[.*\]', clean, re.DOTALL)
        if match:
            clean = match.group(0)
        actions = json.loads(clean)

        normalized = []
        aliases = {
            'SHOOT': 'ATTACK', 'FIRE': 'ATTACK',
            'LEFT': 'MOVE_LEFT', 'RIGHT': 'MOVE_RIGHT',
            'FORWARD': 'MOVE_FORWARD'
        }

        for a in actions:
            a_up = str(a).upper().strip()
            a_up = aliases.get(a_up, a_up)
            if a_up in valid_actions:
                normalized.append(a_up)
        return normalized
    except:
        return []


class TrainingSession:
    """Manages a complete training session with HERON architecture."""

    def __init__(self, scenario, episodes, visible=False, use_helper=True,
                 heron_strategy="initial", save_dir="training_results", use_oracle=True,
                 use_reviewer=True, quiet=False):
        self.scenario = scenario
        self.episodes = episodes
        self.visible = visible
        self.use_helper = use_helper
        self.use_reviewer = use_reviewer
        self.use_oracle = use_oracle
        self.quiet = quiet
        self.heron_strategy = parse_strategy(heron_strategy) if use_helper else None
        self.heron_trainer = HERONTrainer(self.heron_strategy) if use_helper else None

        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        strategy_name = f"_{heron_strategy}" if use_helper else ""
        reviewer_suffix = "_reviewer" if use_reviewer else ""
        self.output_dir = os.path.join(save_dir, f"{scenario}{strategy_name}{reviewer_suffix}_{timestamp}")
        os.makedirs(self.output_dir, exist_ok=True)

        self.rewards_history = []
        self.wins_history = []
        self.helper_suggestions = 0
        self.reviewer_corrections = 0
        self.reviewer_agreements = 0

    def run(self):
        """Execute the training session."""
        if self.use_helper and not check_lmstudio_connection():
            self.use_helper = False
            self.use_reviewer = False

        env = create_vizdoom_env(scenario=self.scenario, visible=self.visible, use_oracle=self.use_oracle)
        agent = DQNCnnAgent(state_shape=env.state_shape, action_size=env.action_size)

        reviewer = None
        if self.use_reviewer:
            try:
                reviewer = ReviewerLLM(
                    model_path=REVIEWER_MODEL_PATH,
                    quantization=REVIEWER_QUANTIZATION,
                    history_size=REVIEWER_HISTORY_SIZE
                )
            except Exception:
                self.use_reviewer = False
                reviewer = None

        try:
            for episode in range(self.episodes):
                state = env.reset()
                done = False
                total_reward = 0
                step = 0

                if self.use_helper:
                    self.heron_trainer.reset_episode()
                if self.use_reviewer and reviewer:
                    reviewer.reset_episode()

                action_queue = deque()

                while not done:
                    should_call = False
                    if self.use_helper and not action_queue:
                        should_call = self.heron_trainer.should_call_helper()

                    if should_call:
                        desc = get_oracle_description(env, None)
                        helper_actions = call_helper_llm(desc, env.action_names, self.scenario)

                        if helper_actions:
                            self.helper_suggestions += 1
                            final_actions = helper_actions

                            if self.use_reviewer and reviewer:
                                try:
                                    corrected_actions = reviewer.review_plan(
                                        state_desc=desc,
                                        helper_plan=helper_actions,
                                        available_actions=env.action_names,
                                        scenario=self.scenario
                                    )

                                    if corrected_actions:
                                        final_actions = corrected_actions

                                        if corrected_actions == helper_actions:
                                            self.reviewer_agreements += 1
                                        else:
                                            self.reviewer_corrections += 1
                                except Exception:
                                    final_actions = helper_actions

                            for a in final_actions:
                                idx = env.get_action_from_name(a)
                                if idx is not None:
                                    action_queue.append(idx)

                    if action_queue:
                        action = action_queue.popleft()
                    else:
                        action = agent.act(state)

                    next_state, reward, done, info = env.step(action)

                    if self.use_reviewer and reviewer:
                        action_name = env.action_names[action] if action < len(env.action_names) else f"ACTION_{action}"
                        state_desc_short = f"HP={state[0]:.0f}, Ammo={state[1]:.0f}" if len(state) >= 2 else "unknown"
                        reviewer.update_history(state_desc_short, action_name, reward)

                    agent.remember(state, action, reward, next_state, done)
                    if len(agent.memory) > agent.batch_size and step % 4 == 0:
                        agent.replay()

                    state = next_state
                    total_reward += reward
                    step += 1

                agent.decay_epsilon()
                self.rewards_history.append(total_reward)
                self.wins_history.append(info.get('victory', False))

                if not self.quiet:
                    print(f"Episode {episode+1}/{self.episodes} | Reward: {total_reward:.1f} | Steps: {step} | Epsilon: {agent.epsilon:.2f}")

                if (episode + 1) % 50 == 0:
                    agent.save(os.path.join(self.output_dir, f"model_ep{episode+1}"))

        except KeyboardInterrupt:
            pass
        finally:
            if 'env' in locals():
                env.close()
            if 'agent' in locals():
                agent.save(os.path.join(self.output_dir, "final_model"))

            if self.use_reviewer and reviewer:
                reviewer.print_stats()
                reviewer.clear_cache()

            self.save_stats()

    def save_stats(self):
        """Save training statistics to JSON file."""
        stats = {
            'rewards': self.rewards_history,
            'wins': [int(x) for x in self.wins_history],
            'helper_suggestions': self.helper_suggestions,
            'reviewer_corrections': self.reviewer_corrections,
            'reviewer_agreements': self.reviewer_agreements,
            'agreement_rate': self.reviewer_agreements / self.helper_suggestions if self.helper_suggestions > 0 else 0.0
        }
        with open(os.path.join(self.output_dir, "stats.json"), "w") as f:
            json.dump(stats, f, indent=2)


def configure_gpu_limits(args):
    """Configure GPU memory limits based on arguments."""
    if args.dqn_cpu:
        os.environ['CUDA_VISIBLE_DEVICES'] = ''
    elif args.dqn_gpu_fraction:
        try:
            import tensorflow as tf
            gpus = tf.config.list_physical_devices('GPU')
            if gpus:
                for gpu in gpus:
                    tf.config.experimental.set_memory_growth(gpu, True)
                    memory_limit_mb = int(8192 * args.dqn_gpu_fraction)
                    tf.config.experimental.set_virtual_device_configuration(
                        gpu,
                        [tf.config.experimental.VirtualDeviceConfiguration(
                            memory_limit=memory_limit_mb)]
                    )
        except Exception:
            pass


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="HERON DQN Training with Helper/Reviewer LLMs")
    parser.add_argument('--scenario', default='defend_the_center',
                       choices=['basic', 'deadly_corridor', 'defend_the_center'])
    parser.add_argument('--episodes', type=int, default=100)
    parser.add_argument('--strategy', default='initial',
                       choices=['initial', 'final_01', 'final_02', 'random'])
    parser.add_argument('--oracle', action='store_true', default=True,
                       help='Use Oracle mode (numerical features)')
    parser.add_argument('--no-helper', action='store_true', default=False,
                       help='Disable Helper LLM (Level 2) - train DQN only')
    parser.add_argument('--no-reviewer', action='store_true', default=False,
                       help='Disable Reviewer (Level 3)')
    parser.add_argument('--quiet', action='store_true', default=False,
                       help='Disable verbose logging')
    parser.add_argument('--dqn-gpu-fraction', type=float, default=None,
                       help='Limit DQN GPU memory (0.0-1.0)')
    parser.add_argument('--dqn-cpu', action='store_true', default=False,
                       help='Force DQN on CPU')

    args = parser.parse_args()
    configure_gpu_limits(args)

    use_helper = not args.no_helper
    use_reviewer = not args.no_reviewer and use_helper  # Reviewer requires Helper

    session = TrainingSession(
        args.scenario,
        args.episodes,
        use_helper=use_helper,
        heron_strategy=args.strategy,
        use_oracle=args.oracle,
        use_reviewer=use_reviewer,
        quiet=args.quiet
    )
    session.run()
