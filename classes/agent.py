from collections import deque
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.models import load_model
import os
import pickle
import random
import numpy as np


class DQNAgent:
    def __init__(self, state_size, action_size, load_model_path):
        self.state_size = state_size
        self.action_size = action_size
        self.memory = deque(maxlen=2000)
        self.gamma = 0.95  # discount rate
        self.epsilon = 1.0 # exploration rate
        self.epsilon_min = 0.01 # 0.05
        self.epsilon_decay = 0.995
        self.learning_rate = 0.001
        self.target_update_freq = 10
        self.train_step = 0
        if load_model_path:
            self.load(load_model_path)
        else:
            self.model = self._build_model()
            self.target_model = self._build_model()

    def _build_model(self):
        model = Sequential()
        model.add(Dense(128, input_dim=self.state_size, activation='relu'))
        model.add(Dense(128, activation='relu'))
        model.add(Dense(64, activation='relu'))
        model.add(Dense(self.action_size, activation='linear'))
        model.compile(loss='mse', optimizer=Adam(learning_rate=self.learning_rate))
        return model

    def update_target_model(self):
        self.target_model.set_weights(self.model.get_weights())

    def load(self, path_prefix):
        self.model = load_model(f"/{path_prefix}.keras") # enter model path
        self.target_model = load_model(f"/{path_prefix}.keras")
        memory_path = f"/{path_prefix}_memory.pkl" # enter model path
        if os.path.exists(memory_path):
            with open(memory_path, 'rb') as f:
                self.memory = pickle.load(f)

    def remember(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))

    def act(self, state, env):
        valid_actions = env.get_valid_actions()
        state = state.reshape(1, -1)
        if np.random.rand() <= self.epsilon:
            return random.choice(valid_actions)
        q_values = self.model.predict(state, verbose=0)[0]
        masked_q = np.full(self.action_size, -np.inf)
        masked_q[valid_actions] = q_values[valid_actions]
        return int(np.argmax(masked_q))


    def replay(self, batch_size, env):
        if len(self.memory) < batch_size:
            return
        minibatch = random.sample(self.memory, batch_size)
        states = []
        targets = []
        for state, action, reward, next_state, done in minibatch:
            state = state.reshape(1, -1)
            next_state = next_state.reshape(1, -1)
            target_q = self.model.predict(state, verbose=0)[0]
            if done:
                target_q[action] = reward
            else:
                valid_actions = env.get_valid_actions()
                next_q = self.target_model.predict(next_state, verbose=0)[0]
                masked_next_q = np.full(self.action_size, -np.inf)
                masked_next_q[valid_actions] = next_q[valid_actions]
                target_q[action] = reward + self.gamma * np.max(masked_next_q)

            states.append(state[0])
            targets.append(target_q)

        self.model.fit(np.array(states), np.array(targets), epochs=1, verbose=0)

        self.train_step += 1
        if self.train_step % self.target_update_freq == 0:
            self.update_target_model()

        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay

    def save(self, path_prefix):
        self.model.save(f"/{path_prefix}.keras") # enter model path
        with open(f"/{path_prefix}_memory.pkl", 'wb') as f: # enter model path
            pickle.dump(self.memory, f)
        with open(f"/{path_prefix}_epsilon.txt", 'w') as f: # enter model path
            f.write(str(self.epsilon))




