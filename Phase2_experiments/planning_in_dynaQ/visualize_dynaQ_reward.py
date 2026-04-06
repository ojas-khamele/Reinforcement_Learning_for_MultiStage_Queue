import os
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# ============================================================
# CONFIG
# ============================================================
BASE_DIR = "runs"
PLANNING_STEPS = [0, 5, 10, 15, 20]

ALGORITHM = "DynaQ"
SEED = 50

# Baseline reference values
BASE_MEAN = 37.34
BASE_STD = 6.83
BASE_MIN = 23.30
BASE_MAX = 48.12


# ============================================================
# LOAD LOG
# ============================================================
def load_log(planning_step):
    log_path = os.path.join(
        BASE_DIR,
        f"dynaQ_{planning_step}",
        "training_log.csv"
    )
    return pd.read_csv(log_path)


# ============================================================
# MOVING AVERAGE
# ============================================================
def moving_average(data, window):
    return data.groupby(data.index // window).mean()


# ============================================================
# BASELINE PLOT
# ============================================================
def plot_baseline(ax, x_points):
    ax.axhline(BASE_MEAN, color="black", linestyle="--", label="Baseline Mean")

    ax.fill_between(
        x_points,
        BASE_MEAN - BASE_STD,
        BASE_MEAN + BASE_STD,
        color="gray",
        alpha=0.2,
        label="Baseline ± STD"
    )

    ax.axhline(BASE_MIN, color="red", linestyle=":", label="Baseline Min")
    ax.axhline(BASE_MAX, color="green", linestyle=":", label="Baseline Max")


# ============================================================
# MAIN PLOT FUNCTION
# ============================================================
def plot_dynaQ_planning_comparison(window):
    plt.figure(figsize=(10, 6))

    max_len = 0

    for p in PLANNING_STEPS:
        df = load_log(p)
        rewards = df["Reward"]
        avg_rewards = moving_average(rewards, window)

        plt.plot(avg_rewards, label=f"Planning Steps = {p}")
        max_len = max(max_len, len(avg_rewards))

    x_points = np.arange(max_len)
    plot_baseline(plt.gca(), x_points)

    plt.title(f"DynaQ Reward Comparison (Seed={SEED})")
    plt.xlabel(f"Episode (averaged over {window})")
    plt.ylabel("Reward")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()


# ============================================================
# RUN
# ============================================================
if __name__ == "__main__":
    window = int(input("Enter averaging window (e.g., 20, 50, 100): "))
    plot_dynaQ_planning_comparison(window)
