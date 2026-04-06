import os
import csv
import pickle
import numpy as np
import gymnasium as gym
from airport_env import AirportEnv
from airport_env_without_limits import AirportEnv as AirportEnvWithoutLimits


# ------------------------------------------------------------
# Convert continuous observation to discrete key (same as train)
# ------------------------------------------------------------
def state_to_key(obs):
    return tuple(int(x) for x in np.round(obs))


# ------------------------------------------------------------
# Greedy Q-action (correct for evaluation)
# ------------------------------------------------------------
def choose_action_greedy(Q, s_key, action_dim):
    if s_key not in Q:
        return np.random.randint(action_dim)  # fallback
    return int(np.argmax(Q[s_key]))


# ------------------------------------------------------------
# Evaluate Q-table
# ------------------------------------------------------------
def evaluate_dynaQ(seed_value, episodes, env_class, max_steps=300):

    model_dir = fr"runs\dynaQ_{seed_value}"
    model_path = os.path.join(model_dir, "Q_ep1000.pkl")

    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Missing Q-table: {model_path}")

    # Load Q-table
    with open(model_path, "rb") as f:
        Q = pickle.load(f)

    # Build environment
    env = env_class()
    env.reset(seed=100)
    env.action_space.seed(100)

    action_dim = env.action_space.n
    rewards = []

    for ep in range(1, episodes + 1):

        obs, _ = env.reset(seed=100 + ep)
        s_key = state_to_key(obs.astype(np.float32))

        total_reward = 0

        for step in range(max_steps):

            action = choose_action_greedy(Q, s_key, action_dim)

            next_obs, reward, terminated, truncated, _ = env.step(action)

            s_key = state_to_key(next_obs.astype(np.float32))
            total_reward += reward

            if terminated or truncated:
                break

        rewards.append(total_reward)

    env.close()
    return np.mean(rewards)


# ------------------------------------------------------------
# Main evaluation grid
# ------------------------------------------------------------
SEEDS = [10, 20, 30, 40, 50]
EPISODE_COUNTS = [50, 100, 500]

ENVIRONMENTS = {
    "WithLimits": AirportEnv,
    "WithoutLimits": AirportEnvWithoutLimits
}

OUTPUT_CSV = "dynaQ_tests.csv"

with open(OUTPUT_CSV, "w", newline="") as f:

    writer = csv.writer(f)
    writer.writerow(["Environment", "Seed", "Episodes", "MeanReward"])

    for env_name, env_class in ENVIRONMENTS.items():

        for seed in SEEDS:

            for eps in EPISODE_COUNTS:

                mean_reward = evaluate_dynaQ(seed, eps, env_class)

                print(
                    f"{env_name} | Seed {seed} | Episodes {eps} → Mean Reward = {mean_reward:.2f}"
                )

                writer.writerow([env_name, seed, eps, round(mean_reward, 4)])

print(f"\n✅ Saved all Dyna-Q evaluation results to {OUTPUT_CSV}")