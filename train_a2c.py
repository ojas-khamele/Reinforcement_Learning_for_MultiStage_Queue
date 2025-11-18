# ==========================================
# train_a2c_final.py (Comparable with R2D2)
# ==========================================
import os
import time
import json
import csv
from datetime import datetime

import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, Model, Input, optimizers
import gymnasium as gym
from airport_env import AirportEnv

# ===========================
# Reproducibility
# ===========================
SEED = 50
np.random.seed(SEED)
tf.random.set_seed(SEED)
import random; random.seed(SEED)


# ===========================
# Config / Hyperparameters
# ===========================
RUN_NAME = datetime.now().strftime("run_a2c_%Y%m%d_%H%M%S")
RUN_DIR = os.path.join("runs", RUN_NAME)
os.makedirs(RUN_DIR, exist_ok=True)

CONFIG = {
    "ALGO": "A2C",
    "GAMMA": 0.99,
    "ACTOR_LR": 0.0005,
    "CRITIC_LR": 0.0015,
    "ENTROPY_COEF": 0.01,
    "VALUE_COEF": 0.5,
    "GRAD_CLIP_NORM": 0.5,
    "EPISODES": 1000,
    "MAX_STEPS": 300,
    "SAVE_EVERY": 10,
    "ILLEGAL_ACTION_PENALTY": -5.0,
}

with open(os.path.join(RUN_DIR, "config.json"), "w") as f:
    json.dump(CONFIG, f, indent=2)

# ===========================
# Environment
# ===========================
env = gym.wrappers.NormalizeObservation(AirportEnv())
env.reset(seed=SEED)
env.action_space.seed(SEED)

state_size = env.observation_space.shape[0]
action_size = env.action_space.n

# ===========================
# Actor & Critic Networks
# ===========================
def build_actor(input_dim, action_dim):
    x = Input(shape=(input_dim,), name="actor_input")
    h = layers.Dense(128, activation="relu")(x)
    h = layers.Dense(128, activation="relu")(h)
    logits = layers.Dense(action_dim, name="logits")(h)
    return Model(x, logits, name="Actor")

def build_critic(input_dim):
    x = Input(shape=(input_dim,), name="critic_input")
    h = layers.Dense(128, activation="relu")(x)
    h = layers.Dense(128, activation="relu")(h)
    value = layers.Dense(1, name="value")(h)
    return Model(x, value, name="Critic")

actor = build_actor(state_size, action_size)
critic = build_critic(state_size)
actor_optimizer = optimizers.Adam(learning_rate=CONFIG["ACTOR_LR"])
critic_optimizer = optimizers.Adam(learning_rate=CONFIG["CRITIC_LR"])

# ===========================
# Helper Functions
# ===========================
def discount_rewards(rewards, gamma):
    discounted = np.zeros_like(rewards, dtype=np.float32)
    running = 0.0
    for t in reversed(range(len(rewards))):
        running = rewards[t] + gamma * running
        discounted[t] = running
    return discounted

def choose_action(logits):
    probs = tf.nn.softmax(logits).numpy()
    action = np.random.choice(len(probs[0]), p=probs[0])
    return int(action)

def is_illegal_action(action, env):
    # Example placeholder check — replace with your environment’s rule
    return hasattr(env, "is_illegal") and env.is_illegal(action)

# ===========================
# CSV Logging Setup (R2D2 style)
# ===========================
LOG_CSV = os.path.join(RUN_DIR, "a2c_training_log.csv")
with open(LOG_CSV, "w", newline="") as f:
    writer = csv.writer(f)
    writer.writerow([
        "Episode", "TotalReward", "AvgReturn", 
        "ActorLoss", "CriticLoss", "Entropy", 
        "IllegalActions", "Duration_s"
    ])

# ===========================
# Training Loop
# ===========================
print(f"🚀 Starting A2C training | logs -> {LOG_CSV}")

for ep in range(1, CONFIG["EPISODES"] + 1):
    start_time = time.time()
    state, _ = env.reset()
    state = state.astype(np.float32)

    done = False
    ep_reward = 0.0
    illegal_count = 0

    states, actions, rewards = [], [], []

    for step in range(CONFIG["MAX_STEPS"]):
        logits = actor(np.array([state], dtype=np.float32))
        action = choose_action(logits)

        # Check for illegal action
        if is_illegal_action(action, env):
            reward = CONFIG["ILLEGAL_ACTION_PENALTY"]
            illegal_count += 1
            next_state, terminated, truncated = state, False, False
        else:
            next_state, reward, terminated, truncated, _ = env.step(action)

        done = bool(terminated or truncated)
        rewards.append(reward)
        states.append(state)
        actions.append(action)
        state = next_state.astype(np.float32)
        ep_reward += reward

        if done:
            break

    # Compute returns and advantages
    if rewards:
        returns = discount_rewards(rewards, CONFIG["GAMMA"])
        states_arr = np.array(states, dtype=np.float32)
        actions_arr = np.array(actions, dtype=np.int32)
        returns_arr = np.array(returns, dtype=np.float32)

        values = critic(states_arr).numpy().squeeze()
        advantages = returns_arr - values
        avg_return = np.mean(returns_arr)

        # Loss computation
        with tf.GradientTape() as tape_a, tf.GradientTape() as tape_c:
            logits_batch = actor(states_arr, training=True)
            neglogp = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=actions_arr, logits=logits_batch)
            policy_loss = tf.reduce_mean(neglogp * tf.stop_gradient(advantages))

            probs_batch = tf.nn.softmax(logits_batch)
            log_probs_batch = tf.nn.log_softmax(logits_batch)
            entropy = -tf.reduce_mean(tf.reduce_sum(probs_batch * log_probs_batch, axis=1))

            actor_loss = policy_loss - CONFIG["ENTROPY_COEF"] * entropy

            values_batch = critic(states_arr, training=True)
            critic_loss = tf.reduce_mean(tf.square(returns_arr.reshape(-1, 1) - values_batch))

        # Gradient update
        actor_grads = tape_a.gradient(actor_loss, actor.trainable_variables)
        critic_grads = tape_c.gradient(critic_loss, critic.trainable_variables)
        actor_grads = [tf.clip_by_norm(g, CONFIG["GRAD_CLIP_NORM"]) if g is not None else None for g in actor_grads]
        critic_grads = [tf.clip_by_norm(g, CONFIG["GRAD_CLIP_NORM"]) if g is not None else None for g in critic_grads]

        actor_optimizer.apply_gradients(zip(actor_grads, actor.trainable_variables))
        critic_optimizer.apply_gradients(zip(critic_grads, critic.trainable_variables))

        duration = time.time() - start_time

        # Logging
        with open(LOG_CSV, "a", newline="") as f:
            writer = csv.writer(f)
            writer.writerow([
                ep, round(ep_reward, 2), round(avg_return, 2),
                float(actor_loss.numpy()), float(critic_loss.numpy()),
                float(entropy.numpy()), illegal_count, round(duration, 2)
            ])

        print(f"🏁 Ep {ep:4d} | R: {ep_reward:7.2f} | A:{actor_loss.numpy():.4f} | C:{critic_loss.numpy():.4f} | E:{entropy.numpy():.4f} | Illegals:{illegal_count} | T:{duration:.2f}s")
    else:
        print(f"⚠️ Ep {ep:4d}: No valid steps (reset anomaly)")

    # Save checkpoints
    if ep % CONFIG["SAVE_EVERY"] == 0:
        actor.save(os.path.join(RUN_DIR, f"actor_ep{ep}.keras"))
        critic.save(os.path.join(RUN_DIR, f"critic_ep{ep}.keras"))
        print(f"💾 Saved models at episode {ep}")

env.close()
print("✅ A2C training finished.")
