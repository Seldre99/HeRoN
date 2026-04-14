"""
DQN Agent for VizDoom with Oracle Mode support.

Supports two operating modes:
- Oracle Mode: MLP network for vectorized numerical inputs
- Visual Mode: CNN network for image-based inputs

"""

import os
import pickle
import random
from collections import deque

import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.layers import Conv2D, Dense, Flatten, Input, Lambda
from tensorflow.keras.models import Model, Sequential, load_model
from tensorflow.keras.optimizers import Adam


def build_dqn_dense(input_shape, action_size, hidden_units=None):
    """
    Build an MLP network for vectorized Oracle inputs.

    Args:
        input_shape: Input dimensions tuple, e.g., (10,)
        action_size: Number of possible actions
        hidden_units: List of hidden layer sizes

    Returns:
        Compiled Keras model
    """
    if hidden_units is None:
        hidden_units = [128, 128]

    model = Sequential([
        Input(shape=input_shape),
        Dense(hidden_units[0], activation='relu'),
        Dense(hidden_units[1], activation='relu'),
        Dense(action_size, activation='linear')
    ])

    model.compile(optimizer=Adam(learning_rate=0.001), loss='mse')
    return model


def build_dqn_cnn(input_shape, action_size):
    """
    Build a CNN network for visual inputs.

    Args:
        input_shape: Input dimensions tuple, e.g., (84, 84, 4)
        action_size: Number of possible actions

    Returns:
        Compiled Keras model
    """
    model = Sequential([
        Conv2D(16, kernel_size=3, strides=1, activation='relu',
               padding='same', input_shape=input_shape),
        keras.layers.MaxPooling2D(pool_size=4),
        Conv2D(16, kernel_size=3, strides=1, activation='relu',
               padding='same'),
        keras.layers.MaxPooling2D(pool_size=4),
        Flatten(),
        Dense(128, activation='relu'),
        Dense(action_size, activation='linear')
    ])

    model.compile(optimizer=Adam(learning_rate=0.001), loss='mse')
    return model


def build_dueling_dqn_cnn(input_shape, action_size):
    """
    Build a Dueling DQN architecture for visual inputs.

    Improves learning stability by separating value and advantage streams.

    Args:
        input_shape: Input dimensions tuple
        action_size: Number of possible actions

    Returns:
        Compiled Keras model
    """
    inputs = Input(shape=input_shape)

    x = Conv2D(32, kernel_size=8, strides=4, activation='relu')(inputs)
    x = Conv2D(64, kernel_size=4, strides=2, activation='relu')(x)
    x = Conv2D(64, kernel_size=3, strides=1, activation='relu')(x)
    x = Flatten()(x)

    value = Dense(256, activation='relu')(x)
    value = Dense(1)(value)

    advantage = Dense(256, activation='relu')(x)
    advantage = Dense(action_size)(advantage)

    def combine_streams(inputs):
        value, advantage = inputs
        return value + (advantage - tf.reduce_mean(advantage, axis=1, keepdims=True))

    q_values = Lambda(combine_streams)([value, advantage])

    model = Model(inputs=inputs, outputs=q_values)
    model.compile(optimizer=Adam(learning_rate=0.0001), loss='mse')
    return model


class ReplayBuffer:
    """Experience Replay Buffer for DQN training."""

    def __init__(self, capacity=100000):
        """
        Args:
            capacity: Maximum buffer size
        """
        self.buffer = deque(maxlen=capacity)

    def push(self, state, action, reward, next_state, done):
        """Add a transition to the buffer."""
        self.buffer.append((state, action, reward, next_state, done))

    def sample(self, batch_size):
        """
        Sample a random batch from the buffer.

        Returns:
            Tuple of numpy arrays (states, actions, rewards, next_states, dones)
        """
        batch = random.sample(self.buffer, batch_size)
        states, actions, rewards, next_states, dones = zip(*batch)

        return (
            np.array(states, dtype=np.float32),
            np.array(actions, dtype=np.int32),
            np.array(rewards, dtype=np.float32),
            np.array(next_states, dtype=np.float32),
            np.array(dones, dtype=np.float32)
        )

    def __len__(self):
        return len(self.buffer)


class DQNAgent:
    """
    DQN Agent for VizDoom with Oracle Mode support.

    Operating modes:
    - Oracle Mode (state_shape = (N,)): Uses MLP
    - Visual Mode (state_shape = (C, H, W)): Uses CNN

    Features: Double DQN, target network, epsilon-greedy exploration.
    """

    def __init__(self, state_shape, action_size, load_model_path=None,
                 use_dueling=False, use_oracle=True):
        """
        Args:
            state_shape: State dimensions, e.g., (10,) for Oracle, (4, 84, 84) for Visual
            action_size: Number of possible actions
            load_model_path: Path to load existing model
            use_dueling: Use Dueling DQN architecture (Visual mode only)
            use_oracle: Use Oracle mode with MLP
        """
        self.state_shape = state_shape
        self.action_size = action_size
        self.use_dueling = use_dueling
        self.use_oracle = use_oracle

        is_vector_input = len(state_shape) == 1

        if is_vector_input:
            self.use_oracle = True
            self.state_shape_model = state_shape
        else:
            self.use_oracle = False
            if len(state_shape) == 3:
                self.state_shape_model = (state_shape[1], state_shape[2], state_shape[0])
            else:
                self.state_shape_model = state_shape

        self._configure_gpu()

        # Hyperparameters
        self.gamma = 0.99
        self.epsilon = 1.0
        self.epsilon_min = 0.05
        self.epsilon_decay = 0.995
        self.learning_rate = 0.001
        self.batch_size = 32
        self.target_update_freq = 500

        # Build networks
        if self.use_oracle:
            self.policy_net = build_dqn_dense(self.state_shape_model, action_size)
            self.target_net = build_dqn_dense(self.state_shape_model, action_size)
        else:
            if use_dueling:
                self.policy_net = build_dueling_dqn_cnn(self.state_shape_model, action_size)
                self.target_net = build_dueling_dqn_cnn(self.state_shape_model, action_size)
            else:
                self.policy_net = build_dqn_cnn(self.state_shape_model, action_size)
                self.target_net = build_dqn_cnn(self.state_shape_model, action_size)

        self.target_net.set_weights(self.policy_net.get_weights())
        self.optimizer = keras.optimizers.Adam(learning_rate=self.learning_rate)
        self.memory = ReplayBuffer(capacity=100000)
        self.steps_done = 0
        self.last_loss = 0.0

        if load_model_path:
            self.load(load_model_path)

    def _configure_gpu(self):
        """Configure GPU memory settings."""
        gpus = tf.config.list_physical_devices('GPU')
        if gpus:
            try:
                virtual_devices = tf.config.experimental.get_virtual_device_configuration(gpus[0])
                if not virtual_devices:
                    for gpu in gpus:
                        tf.config.experimental.set_memory_growth(gpu, True)
            except (RuntimeError, ValueError):
                pass

    def _preprocess_state(self, state):
        """
        Preprocess state for model input.

        For Oracle: no transformation needed
        For Visual: convert from (C, H, W) to (H, W, C)
        """
        if self.use_oracle:
            return state

        if len(state.shape) == 3:
            return np.transpose(state, (1, 2, 0))
        elif len(state.shape) == 4:
            return np.transpose(state, (0, 2, 3, 1))
        return state

    def act(self, state, valid_actions=None):
        """
        Select an action using epsilon-greedy policy.

        Args:
            state: Current state
            valid_actions: List of valid action indices

        Returns:
            Selected action index
        """
        if valid_actions is None:
            valid_actions = list(range(self.action_size))

        if random.random() < self.epsilon:
            return random.choice(valid_actions)

        state_processed = self._preprocess_state(state)
        state_batch = np.expand_dims(state_processed, axis=0)
        q_values = self.policy_net.predict(state_batch, verbose=0)[0]

        if len(valid_actions) < self.action_size:
            masked_q = np.full_like(q_values, -np.inf)
            masked_q[valid_actions] = q_values[valid_actions]
            return np.argmax(masked_q)

        return np.argmax(q_values)

    def remember(self, state, action, reward, next_state, done):
        """Store a transition in the replay buffer."""
        self.memory.push(state, action, reward, next_state, done)

    def replay(self, batch_size=None):
        """
        Perform a training step using experience replay.

        Args:
            batch_size: Batch size (default: self.batch_size)

        Returns:
            Training loss
        """
        if batch_size is None:
            batch_size = self.batch_size

        if len(self.memory) < batch_size:
            return 0

        states, actions, rewards, next_states, dones = self.memory.sample(batch_size)
        states = self._preprocess_state(states)
        next_states = self._preprocess_state(next_states)

        current_q_values = self.policy_net.predict(states, verbose=0)
        next_q_policy = self.policy_net.predict(next_states, verbose=0)
        next_q_target = self.target_net.predict(next_states, verbose=0)

        best_actions = np.argmax(next_q_policy, axis=1)
        next_q = next_q_target[np.arange(batch_size), best_actions]
        target_q = rewards + (1 - dones) * self.gamma * next_q

        target_q_values = current_q_values.copy()
        target_q_values[np.arange(batch_size), actions] = target_q

        history = self.policy_net.fit(states, target_q_values,
                                       batch_size=batch_size,
                                       epochs=1, verbose=0)
        loss = history.history['loss'][0]
        self.last_loss = loss

        self.steps_done += 1
        if self.steps_done % self.target_update_freq == 0:
            self.target_net.set_weights(self.policy_net.get_weights())

        return loss

    def simple_imitation_learning(self, states, helper_actions, weight=0.5):
        """
        Behavioral cloning from Oracle/Helper actions.

        Args:
            states: List of states
            helper_actions: List of action indices from Helper/Oracle
            weight: BC loss weight

        Returns:
            Total loss
        """
        if not states or not helper_actions or len(states) < 4:
            return 0.0

        states_array = np.array([self._preprocess_state(np.array(s)) for s in states])
        actions_tensor = tf.constant(helper_actions, dtype=tf.int32)

        with tf.GradientTape() as tape:
            q_values = self.policy_net(states_array, training=True)
            bc_loss = tf.nn.sparse_softmax_cross_entropy_with_logits(
                labels=actions_tensor,
                logits=q_values
            )
            mean_bc_loss = tf.reduce_mean(bc_loss) * 10.0

        variables = self.policy_net.trainable_variables
        gradients = tape.gradient(mean_bc_loss, variables)
        self.optimizer.apply_gradients(zip(gradients, variables))

        return float(mean_bc_loss)

    def decay_epsilon(self):
        """Decay epsilon for reduced exploration over time."""
        self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)

    def save(self, path):
        """
        Save model and agent state.

        Args:
            path: Base path for saving (without extension)
        """
        dir_path = os.path.dirname(path) if os.path.dirname(path) else '.'
        os.makedirs(dir_path, exist_ok=True)

        self.policy_net.save(f"{path}_policy.keras")
        self.target_net.save(f"{path}_target.keras")

        agent_state = {
            'epsilon': self.epsilon,
            'steps_done': self.steps_done,
            'state_shape': self.state_shape,
            'action_size': self.action_size,
            'use_dueling': self.use_dueling,
            'use_oracle': self.use_oracle
        }
        with open(f"{path}_state.pkl", 'wb') as f:
            pickle.dump(agent_state, f)

        with open(f"{path}_memory.pkl", 'wb') as f:
            pickle.dump(self.memory, f)

    def load(self, path):
        """
        Load model and agent state.

        Args:
            path: Base path of the model (without extension)
        """
        self.policy_net = load_model(f"{path}_policy.keras")
        self.target_net = load_model(f"{path}_target.keras")

        with open(f"{path}_state.pkl", 'rb') as f:
            agent_state = pickle.load(f)

        self.epsilon = agent_state['epsilon']
        self.steps_done = agent_state['steps_done']

        memory_path = f"{path}_memory.pkl"
        if os.path.exists(memory_path):
            with open(memory_path, 'rb') as f:
                self.memory = pickle.load(f)


# Alias for backwards compatibility
DQNCnnAgent = DQNAgent
