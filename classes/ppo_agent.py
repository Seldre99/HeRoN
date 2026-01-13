import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, Model, optimizers
import os
from tensorflow.keras.models import load_model


class PPOAgent:
    def __init__(
        self,
        state_size,
        action_size,
        gamma=0.99,
        lam=0.95,
        clip_ratio=0.2,
        lr=3e-4
    ):
        self.state_size = state_size
        self.action_size = action_size
        self.gamma = gamma
        self.lam = lam
        self.clip_ratio = clip_ratio

        self.actor = self._build_actor()
        self.critic = self._build_critic()

        self.actor_opt = optimizers.Adam(lr)
        self.critic_opt = optimizers.Adam(lr)

    def _build_actor(self):
        inputs = layers.Input(shape=(self.state_size,))
        x = layers.Dense(128, activation="relu")(inputs)
        x = layers.Dense(128, activation="relu")(x)
        x = layers.Dense(64, activation="relu")(x)
        outputs = layers.Dense(self.action_size, activation="softmax")(x)
        return Model(inputs, outputs)

    def _build_critic(self):
        inputs = layers.Input(shape=(self.state_size,))
        x = layers.Dense(128, activation="relu")(inputs)
        x = layers.Dense(128, activation="relu")(x)
        x = layers.Dense(64, activation="relu")(x)
        outputs = layers.Dense(1)(x)
        return Model(inputs, outputs)

    def act(self, state, action_mask):
        probs = self.actor(state).numpy()[0]
        probs = np.nan_to_num(probs, nan=0.0, posinf=0.0, neginf=0.0)
        probs = probs * action_mask

        total = np.sum(probs)

        if total <= 1e-8:
            valid_actions = np.where(action_mask == 1)[0]
            action = np.random.choice(valid_actions)
            log_prob = 0.0
            return action, log_prob

        probs /= total

        action = np.random.choice(self.action_size, p=probs)
        log_prob = np.log(probs[action] + 1e-8)

        return action, log_prob

    def compute_gae(self, rewards, values, dones):
        advantages = []
        gae = 0
        values = np.append(values, 0)

        for t in reversed(range(len(rewards))):
            delta = rewards[t] + self.gamma * values[t + 1] * (1 - dones[t]) - values[t]
            gae = delta + self.gamma * self.lam * (1 - dones[t]) * gae
            advantages.insert(0, gae)

        return np.array(advantages)

    def train(self, states, actions, old_log_probs, returns, advantages):
        actions = np.array(actions)

        with tf.GradientTape() as tape:
            probs = self.actor(states)
            action_probs = tf.reduce_sum(
                probs * tf.one_hot(actions, self.action_size), axis=1
            )

            ratios = tf.exp(tf.math.log(action_probs + 1e-8) - old_log_probs)
            clipped = tf.clip_by_value(
                ratios, 1 - self.clip_ratio, 1 + self.clip_ratio
            )

            loss = -tf.reduce_mean(
                tf.minimum(ratios * advantages, clipped * advantages)
            )

        grads = tape.gradient(loss, self.actor.trainable_variables)
        self.actor_opt.apply_gradients(zip(grads, self.actor.trainable_variables))

        with tf.GradientTape() as tape:
            values = self.critic(states)
            value_loss = tf.reduce_mean((returns - tf.squeeze(values)) ** 2)

        grads = tape.gradient(value_loss, self.critic.trainable_variables)
        self.critic_opt.apply_gradients(zip(grads, self.critic.trainable_variables))

    def act_deterministic(self, state, action_mask):
        probs = self.actor(state).numpy()[0]
        probs = probs * action_mask
        probs /= np.sum(probs)
        return np.argmax(probs)

    def save(self, path):
        os.makedirs(path, exist_ok=True)
        self.actor.save(f"{path}/actor.keras")
        self.critic.save(f"{path}/critic.keras")

    def load(self, path):
        self.actor = load_model(f"{path}/actor.keras")
        self.critic = load_model(f"{path}/critic.keras")
