import os
import csv
import numpy as np
import tensorflow as tf
import gymnasium as gym
from airport_env import AirportEnv
from airport_env_without_limits import AirportEnv as AirportEnvWithoutLimits

# --------------------------------------
# Rebuild architecture
# --------------------------------------
def build_actor(input_dim, action_dim):
    inp = tf.keras.Input(shape=(input_dim,))
    x = tf.keras.layers.Dense(128, activation="relu")(inp)
    x = tf.keras.layers.Dense(128, activation="relu")(x)
    logits = tf.keras.layers.Dense(action_dim)(x)
    return tf.keras.Model(inp, logits)

def build_critic(input_dim):
    inp = tf.keras.Input(shape=(input_dim,))
    x = tf.keras.layers.Dense(128, activation="relu")(inp)
    x = tf.keras.layers.Dense(128, activation="relu")(x)
    value = tf.keras.layers.Dense(1)(x)
    return tf.keras.Model(inp, value)

# --------------------------------------
# Helper for action selection
# --------------------------------------
def choose_action(logits):
    probs = tf.nn.softmax(logits).numpy()[0]
    return int(np.random.choice(len(probs), p=probs))


# --------------------------------------
# Evaluation function
# --------------------------------------
def evaluate_model(seed_value, episodes, env_class, max_steps=300):

    model_dir = fr"runs\run_a2c_{seed_value}"
    actor_path = os.path.join(model_dir, "actor_ep1000.keras")
    critic_path = os.path.join(model_dir, "critic_ep1000.keras")

    env = gym.wrappers.NormalizeObservation(env_class())
    env.reset(seed=100)
    env.action_space.seed(100)

    state_size = env.observation_space.shape[0]
    action_size = env.action_space.n

    # Load models
    actor = build_actor(state_size, action_size)
    critic = build_critic(state_size)

    actor.load_weights(actor_path)
    critic.load_weights(critic_path)

    rewards = []

    for ep in range(1, episodes + 1):

        state, _ = env.reset(seed=100 + ep)
        state = state.astype(np.float32)

        total_reward = 0

        for step in range(max_steps):

            logits = actor(np.array([state], dtype=np.float32))
            action = choose_action(logits)

            next_state, reward, terminated, truncated, _ = env.step(action)

            state = next_state.astype(np.float32)
            total_reward += reward

            if terminated or truncated:
                break

        rewards.append(total_reward)

    env.close()
    return np.mean(rewards)


# --------------------------------------
# Main loop
# --------------------------------------
SEEDS = [10, 20, 30, 40, 50]
EPISODE_COUNTS = [50, 100, 500]

ENVIRONMENTS = {
    "WithLimits": AirportEnv,
    "WithoutLimits": AirportEnvWithoutLimits
}

OUTPUT_CSV = "a2c_tests.csv"

with open(OUTPUT_CSV, "w", newline="") as f:

    writer = csv.writer(f)
    writer.writerow(["Environment", "Seed", "Episodes", "MeanReward"])

    for env_name, env_class in ENVIRONMENTS.items():

        for seed in SEEDS:

            for eps in EPISODE_COUNTS:

                mean_reward = evaluate_model(seed, eps, env_class)

                print(
                    f"{env_name} | Seed {seed} | Episodes {eps} → Mean Reward = {mean_reward:.2f}"
                )

                writer.writerow([env_name, seed, eps, round(mean_reward, 4)])

print(f"\n✅ Saved all results to {OUTPUT_CSV}")