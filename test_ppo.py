import os
import csv
import numpy as np
import tensorflow as tf
import gymnasium as gym
from airport_env import AirportEnv


# -------------------------------------------------------
# Rebuild PPO Actor & Critic (must match training script)
# -------------------------------------------------------
def build_actor(input_dim, action_dim):
    inputs = tf.keras.Input(shape=(input_dim,))
    x = tf.keras.layers.Dense(128, activation='relu')(inputs)
    x = tf.keras.layers.Dense(128, activation='relu')(x)
    outputs = tf.keras.layers.Dense(action_dim, activation='softmax')(x)
    return tf.keras.Model(inputs, outputs)


def build_critic(input_dim):
    inputs = tf.keras.Input(shape=(input_dim,))
    x = tf.keras.layers.Dense(128, activation='relu')(inputs)
    x = tf.keras.layers.Dense(128, activation='relu')(x)
    outputs = tf.keras.layers.Dense(1)(x)
    return tf.keras.Model(inputs, outputs)


# -------------------------------------------------------
# Greedy action selection (correct for PPO evaluation)
# -------------------------------------------------------
def choose_action(actor, state):
    state = np.expand_dims(state, axis=0)      # (1, state_dim)
    probs = actor(state, training=False).numpy()[0]
    return int(np.argmax(probs))               # greedy


# -------------------------------------------------------
# Evaluation function (same structure as A2C version)
# -------------------------------------------------------
def evaluate_model(seed_value, episodes, max_steps=300):
    model_dir = fr"runs\SEED_{seed_value}_ppo"
    actor_path = os.path.join(model_dir, "actor_ep1000.keras")
    critic_path = os.path.join(model_dir, "critic_ep1000.keras")

    # Build env
    env = gym.wrappers.NormalizeObservation(AirportEnv())
    env.reset(seed=100)
    env.action_space.seed(100)

    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.n

    # Load models
    actor = build_actor(state_dim, action_dim)
    critic = build_critic(state_dim)

    actor.load_weights(actor_path)
    critic.load_weights(critic_path)

    rewards = []

    for ep in range(1, episodes + 1):
        state, _ = env.reset(seed=100 + ep)
        state = state.astype(np.float32)

        total_reward = 0

        for step in range(max_steps):
            action = choose_action(actor, state)
            next_state, reward, terminated, truncated, _ = env.step(action)

            state = next_state.astype(np.float32)
            total_reward += reward

            if terminated or truncated:
                break

        rewards.append(total_reward)

    env.close()
    return np.mean(rewards)


# -------------------------------------------------------
# Main loop: evaluate SEED × EPISODES grid
# -------------------------------------------------------
SEEDS = [10, 20, 30, 40, 50]
EPISODE_COUNTS = [50, 100, 500]

OUTPUT_CSV = "ppo_tests.csv"

with open(OUTPUT_CSV, "w", newline="") as f:
    writer = csv.writer(f)
    writer.writerow(["Seed", "Episodes", "MeanReward"])

    for seed in SEEDS:
        for eps in EPISODE_COUNTS:
            mean_reward = evaluate_model(seed, eps)
            print(f"Seed {seed} | Episodes {eps} → Mean Reward = {mean_reward:.2f}")
            writer.writerow([seed, eps, round(mean_reward, 4)])

print(f"\n✅ Saved all PPO evaluation results to {OUTPUT_CSV}")
