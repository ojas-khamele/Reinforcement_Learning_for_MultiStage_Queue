import numpy as np
import random
import csv
from airport_env import AirportEnv

# ============================================================
# Configurations
# ============================================================
SEEDS = [10, 20, 30, 40, 50]
N_RUNS = 1000
N_STEPS = 300
CSV_FILE = "baseline_results_summary.csv"

# ============================================================
# Prepare CSV file
# ============================================================
with open(CSV_FILE, mode='w', newline='') as file:
    writer = csv.writer(file)
    writer.writerow([
        "Seed", "Total Runs", "Max Reward", "Min Reward",
        "Average Reward", "Std Deviation", "25th Percentile",
        "50th Percentile", "75th Percentile"
    ])

# ============================================================
# Run experiments for each SEED
# ============================================================
for SEED in SEEDS:
    np.random.seed(SEED)
    random.seed(SEED)

    env = AirportEnv()
    env.reset(seed=SEED)

    cumulative_rewards = []

    print(f"\n================ Running for SEED = {SEED} ================\n")

    for run in range(1, N_RUNS + 1):
        obs, info = env.reset(seed=SEED + run)
        total_reward = 0.0

        for step in range(N_STEPS):
            action = 0  # fixed action
            obs, reward, done, truncated, info = env.step(action)
            total_reward += reward
            if done or truncated:
                break

        cumulative_rewards.append(total_reward)
        print(f"Run {run:4d} | Cumulative Reward = {total_reward:.2f}")

    # ============================================================
    # Compute statistics including percentiles
    # ============================================================
    cumulative_rewards = np.array(cumulative_rewards)
    max_reward = np.max(cumulative_rewards)
    min_reward = np.min(cumulative_rewards)
    avg_reward = np.mean(cumulative_rewards)
    std_reward = np.std(cumulative_rewards)
    p25 = np.percentile(cumulative_rewards, 25)
    p50 = np.percentile(cumulative_rewards, 50)
    p75 = np.percentile(cumulative_rewards, 75)

    # ============================================================
    # Print summary for this SEED
    # ============================================================
    print("\n================ Summary ================")
    print(f"🌱 SEED: {SEED}")
    print(f"✅ Total Runs: {N_RUNS}")
    print(f"🏆 Max Cumulative Reward: {max_reward:.2f}")
    print(f"⚠️  Min Cumulative Reward: {min_reward:.2f}")
    print(f"📊 Average Cumulative Reward: {avg_reward:.2f}")
    print(f"📈 Std. Deviation: {std_reward:.2f}")
    print(f"📌 25th Percentile: {p25:.2f}")
    print(f"📌 50th Percentile (Median): {p50:.2f}")
    print(f"📌 75th Percentile: {p75:.2f}")
    print("=========================================\n")

    # ============================================================
    # Save summary to CSV
    # ============================================================
    with open(CSV_FILE, mode='a', newline='') as file:
        writer = csv.writer(file)
        writer.writerow([SEED, N_RUNS, max_reward, min_reward, avg_reward,
                         std_reward, p25, p50, p75])

    env.close()

print(f"✅ All results saved to '{CSV_FILE}' successfully.")
