import os
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# ============================================================
# CONFIG
# ============================================================
LOG_DIR = "runs"

ALGORITHM = "dyna"
SEED = [10, 20, 30, 40, 50]

LOSS_COL = "TD_Error"

# ============================================================
# LOAD LOG FILE
# ============================================================
def load_log(k):
    file_path = os.path.join(
        LOG_DIR, f"dynaQ_airport_log_{k}.csv"
    )
    return pd.read_csv(file_path)


# ============================================================
# MOVING AVERAGE
# ============================================================
def moving_average(data, window):
    return data.groupby(data.index // window).mean()


# ============================================================
# MODE 1: Compare across planning steps
# ============================================================
def plot_across_seeds(window=50):
    plt.figure(figsize=(10, 6))

    for k in SEED:
        df = load_log(k)
        avg_loss = moving_average(df[LOSS_COL], window)
        plt.plot(avg_loss, label=f"seed = {k}")

    plt.title("Dyna-Q TD Error Across Seeds")
    plt.xlabel(f"Training Steps (averaged every {window})")
    plt.ylabel("TD Error")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()


# ============================================================
# MODE 2: Single planning step
# ============================================================
def plot_single_k(k, window=50):
    plt.figure(figsize=(10, 6))

    df = load_log(k)
    avg_loss = moving_average(df[LOSS_COL], window)
    plt.plot(avg_loss, label=f"seed = {k}")

    plt.title(f"Dyna-Q TD Error (Seed = {k})")
    plt.xlabel(f"Training Steps (averaged every {window})")
    plt.ylabel("TD Error")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()


# ============================================================
# MAIN LOOP
# ============================================================
while True:
    print("\n===================================")
    print(" Dyna-Q Planning Step Analysis Tool")
    print("===================================\n")
    print("Choose plotting mode:")
    print("1 → Compare across Seeds")
    print("2 → Plot a single graph for seed")
    print("q → Quit")
    
    choice = input("\nEnter choice (1, 2 or q): ").lower()

    if choice == "q":
        print("Exiting tool. Goodbye!\n")
        break

    if choice not in ["1", "2"]:
        print("Invalid choice! Try again.\n")
        continue

    window = int(input("Enter averaging window (e.g., 20, 50, 100): "))

    if choice == "1":
        print("\nPlotting comparison across Seed Values...\n")
        plot_across_seeds(window)

    elif choice == "2":
        k = int(input("Enter Seed Value (10, 20, 30, 40, 50): "))
        print("\nPlotting TD Error...\n")
        plot_single_k(k, window)