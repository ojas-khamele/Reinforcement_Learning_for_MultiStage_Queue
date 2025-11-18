# dyna_q_airport.py
import os
import csv
import time
import random
import pickle
import json
import numpy as np
from airport_env import AirportEnv  # your provided class

SEED = 40
np.random.seed(SEED)
random.seed(SEED)

# ====================================================
# Environment
# ====================================================
env = AirportEnv()
env.reset(seed=SEED)
env.action_space.seed(SEED)

# ====================================================
# Hyperparameters
# ====================================================
N_EPISODES = 1000
N_PLANNING = 7
ALPHA = 0.05
GAMMA = 0.99
EPSILON_START = 1.0
EPSILON_MIN = 0.05
EPSILON_DECAY = 0.996

# ====================================================
# Config Saving (basic)
# ====================================================
run_dir = f"runs/dynaQ_airport_{time.strftime('%Y%m%d_%H%M%S')}"
os.makedirs(run_dir, exist_ok=True)

CONFIG = {
    "ALGO": "Dyna-Q",
    "N_EPISODES": N_EPISODES,
    "N_PLANNING": N_PLANNING,
    "ALPHA": ALPHA,
    "GAMMA": GAMMA,
    "EPSILON_START": EPSILON_START,
    "EPSILON_MIN": EPSILON_MIN,
    "EPSILON_DECAY": EPSILON_DECAY,
    "SEED": SEED
}

with open(os.path.join(run_dir, "config.json"), "w") as f:
    json.dump(CONFIG, f, indent=2)

# ====================================================
# State processing helpers
# ====================================================
def state_to_key(obs):
    """Convert observation (np.array) to hashable discrete key."""
    return tuple(int(x) for x in np.round(obs))

# ====================================================
# Structures
# ====================================================
Q = {}
Model = {}
visited_state_action = []
action_dim = env.action_space.n

def ensure_state(s_key):
    if s_key not in Q:
        Q[s_key] = np.zeros(action_dim)

def choose_action(s_key, eps):
    ensure_state(s_key)
    if random.random() < eps:
        return random.randrange(action_dim)
    return int(np.argmax(Q[s_key]))

def update_q(s, a, r, s_next):
    ensure_state(s)
    ensure_state(s_next)
    td_target = r + GAMMA * np.max(Q[s_next])
    td_error = td_target - Q[s][a]
    Q[s][a] += ALPHA * td_error
    return td_error  # return TD error for tracking

# ====================================================
# Run setup
# ====================================================
log_file = os.path.join(run_dir, "dynaQ_airport_log.csv")

with open(log_file, "w", newline="") as f:
    csv.writer(f).writerow(["Episode", "Reward", "Epsilon", "Duration_s", "VisitedPairs", "TD_Error"])

# ====================================================
# Training loop
# ====================================================
epsilon = EPSILON_START

for ep in range(1, N_EPISODES + 1):
    start = time.time()
    obs, _ = env.reset()
    s_key = state_to_key(obs)
    total_reward = 0
    done = False
    steps = 0
    td_errors = []  # store per-episode TD errors

    while not done:
        steps += 1
        action = choose_action(s_key, epsilon)
        next_obs, reward, done, truncated, _ = env.step(action)
        s_next_key = state_to_key(next_obs)
        total_reward += reward

        # Direct update + collect TD error
        td_error = update_q(s_key, action, reward, s_next_key)
        td_errors.append(abs(td_error))  # absolute error magnitude

        # Update model
        Model[(s_key, action)] = (reward, s_next_key)
        if (s_key, action) not in visited_state_action:
            visited_state_action.append((s_key, action))

        # Planning
        for _ in range(N_PLANNING):
            s_sim, a_sim = random.choice(visited_state_action)
            r_sim, s_next_sim = Model[(s_sim, a_sim)]
            _ = update_q(s_sim, a_sim, r_sim, s_next_sim)  # we can ignore simulated errors for now

        s_key = s_next_key

    epsilon = max(EPSILON_MIN, epsilon * EPSILON_DECAY)
    duration = time.time() - start
    avg_td_error = np.mean(td_errors) if td_errors else 0.0

    # Log progress
    with open(log_file, "a", newline="") as f:
        csv.writer(f).writerow([
            ep,
            round(total_reward, 3),
            round(epsilon, 3),
            round(duration, 2),
            len(visited_state_action),
            round(avg_td_error, 6)
        ])

    print(
        f"🏁 Ep {ep:4d} | Reward: {total_reward:8.3f} | eps={epsilon:.3f} "
        f"| TD_err={avg_td_error:.5f} | t={duration:.2f}s | visited={len(visited_state_action)}"
    )

    # 💾 Save model every 10 episodes
    if ep % 10 == 0:
        with open(os.path.join(run_dir, f"Q_ep{ep}.pkl"), "wb") as f:
            pickle.dump(Q, f)
        with open(os.path.join(run_dir, f"Model_ep{ep}.pkl"), "wb") as f:
            pickle.dump(Model, f)
        print(f"💾 Saved Q & Model at episode {ep}")

# ====================================================
# Wrap up
# ====================================================
print(f"\n✅ Dyna-Q training finished. Logs saved to {log_file}")
env.close()
