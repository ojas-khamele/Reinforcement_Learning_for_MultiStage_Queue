import os
import csv
import numpy as np
import tensorflow as tf
import gymnasium as gym
from airport_env import AirportEnv


# -------------------------------------------------------
# Rebuild R2D2 Q-network (MUST match train architecture)
# -------------------------------------------------------
def build_r2d2_qnetwork(input_dim, action_dim, lstm_units=128):
    x_in = tf.keras.Input(shape=(None, input_dim))     # (batch, seq_len, features)
    lstm_out = tf.keras.layers.LSTM(lstm_units, return_sequences=True)(x_in)
    dense = tf.keras.layers.TimeDistributed(tf.keras.layers.Dense(128, activation='relu'))(lstm_out)
    q_vals = tf.keras.layers.TimeDistributed(tf.keras.layers.Dense(action_dim))(dense)
    return tf.keras.Model(x_in, q_vals)


# -------------------------------------------------------
# Greedy action selection
# -------------------------------------------------------
def choose_action_r2d2(q_network, state):
    """
    state shape required: (1, 1, state_dim)
    """
    inp = np.expand_dims(np.expand_dims(state, axis=0), axis=0)
    q_values = q_network(inp).numpy()[0, 0]              # (num_actions,)
    return int(np.argmax(q_values))


# -------------------------------------------------------
# Evaluation function
# -------------------------------------------------------
def evaluate_model(seed_value, episodes, max_steps=300):
    model_dir = fr"runs\SEED_{seed_value}_r2d2"
    model_path = os.path.join(model_dir, "qnet_ep1000.keras")

    # Env
    env = gym.wrappers.NormalizeObservation(AirportEnv())
    env.reset(seed=100)
    env.action_space.seed(100)

    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.n

    # Rebuild network
    q_network = build_r2d2_qnetwork(state_dim, action_dim)
    q_network.load_weights(model_path)

    rewards = []

    for ep in range(1, episodes + 1):
        state, _ = env.reset(seed=100 + ep)
        state = state.astype(np.float32)

        total_reward = 0

        for step in range(max_steps):
            action = choose_action_r2d2(q_network, state)
            next_state, reward, terminated, truncated, _ = env.step(action)

            state = next_state.astype(np.float32)
            total_reward += reward

            if terminated or truncated:
                break

        rewards.append(total_reward)

    env.close()
    return np.mean(rewards)


# -------------------------------------------------------
# Main loop: test all seeds × episode counts
# -------------------------------------------------------
SEEDS = [10, 20, 30, 40, 50]
EPISODE_COUNTS = [50, 100, 500]

OUTPUT_CSV = "r2d2_tests.csv"

with open(OUTPUT_CSV, "w", newline="") as f:
    writer = csv.writer(f)
    writer.writerow(["Seed", "Episodes", "MeanReward"])

    for seed in SEEDS:
        for eps in EPISODE_COUNTS:
            mean_reward = evaluate_model(seed, eps)
            print(f"Seed {seed} | Episodes {eps} → Mean Reward = {mean_reward:.2f}")
            writer.writerow([seed, eps, round(mean_reward, 4)])

print(f"\n✅ Saved results to {OUTPUT_CSV}")
