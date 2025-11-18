# train_r2d2_final_fixed_stable.py
import os, time, json, csv, random
from datetime import datetime
from collections import deque, namedtuple
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, Model, Input, optimizers
from airport_env import AirportEnv

SEED = 50
np.random.seed(SEED)
random.seed(SEED)
tf.random.set_seed(SEED)


# ============================================================
# ⚙️ Config
# ============================================================
RUN_NAME = datetime.now().strftime("run_r2d2_%Y%m%d_%H%M%S")
RUN_DIR = os.path.join("runs", RUN_NAME)
os.makedirs(RUN_DIR, exist_ok=True)

CONFIG = {
    "ALGO": "R2D2",
    "GAMMA": 0.99,
    "LR": 1e-4,
    "BATCH_SIZE": 32,
    "SEQ_LEN": 8,
    "REPLAY_SIZE": 50000,
    "EPISODES": 1000,
    "MAX_STEPS": 300,
    "SAVE_EVERY": 20,
    "EPS_START": 1.0,
    "EPS_END": 0.05,
    "EPS_DECAY": 0.995,
    "SOFT_TAU": 0.01,
    "SEQ_OVERLAP": 4,
    "GRAD_CLIP": 1.0,
    "SEED": SEED
}

# Save config
with open(os.path.join(RUN_DIR, "config.json"), "w") as f:
    json.dump(CONFIG, f, indent=2)

LOG_CSV = os.path.join(RUN_DIR, "r2d2_training_log.csv")

env = AirportEnv()
env.reset(seed=SEED)
env.action_space.seed(SEED)

state_size = env.observation_space.shape[0]
action_size = env.action_space.n

# ============================================================
# 🧠 Replay Buffer
# ============================================================
Transition = namedtuple('Transition', ('state', 'action', 'reward', 'next_state', 'done'))
class ReplayBuffer:
    def __init__(self, capacity=CONFIG["REPLAY_SIZE"]):
        self.buffer = deque(maxlen=capacity)

    def push(self, *args):
        self.buffer.append(Transition(*args))

    def sample(self, batch_size=CONFIG["BATCH_SIZE"], seq_len=CONFIG["SEQ_LEN"]):
        idxs = np.random.randint(0, len(self.buffer) - seq_len, size=batch_size)
        seqs = []
        for i in idxs:
            seq = list(self.buffer)[i:i+seq_len]
            seqs.append(seq)
        states = np.array([[t.state for t in s] for s in seqs], dtype=np.float32)
        actions = np.array([[t.action for t in s] for s in seqs], dtype=np.int32)
        rewards = np.array([[t.reward for t in s] for s in seqs], dtype=np.float32)
        next_states = np.array([[t.next_state for t in s] for s in seqs], dtype=np.float32)
        dones = np.array([[t.done for t in s] for s in seqs], dtype=np.float32)
        return states, actions, rewards, next_states, dones

    def __len__(self): return len(self.buffer)

replay_buffer = ReplayBuffer()

# ============================================================
# 🧩 Q-Network (LSTM)
# ============================================================
def build_r2d2_qnetwork(input_dim, action_dim, lstm_units=128):
    x_in = Input(shape=(None, input_dim))
    lstm_out = layers.LSTM(lstm_units, return_sequences=True)(x_in)
    dense = layers.TimeDistributed(layers.Dense(128, activation='relu'))(lstm_out)
    q_vals = layers.TimeDistributed(layers.Dense(action_dim))(dense)
    return Model(x_in, q_vals)

q_network = build_r2d2_qnetwork(state_size, action_size)
target_network = build_r2d2_qnetwork(state_size, action_size)
target_network.set_weights(q_network.get_weights())

optimizer = optimizers.Adam(CONFIG["LR"])
loss_fn = tf.keras.losses.MeanSquaredError()

# ============================================================
# 🔧 Helpers
# ============================================================
def soft_update(target, source, tau=CONFIG["SOFT_TAU"]):
    target_weights = target.get_weights()
    source_weights = source.get_weights()
    for i in range(len(target_weights)):
        target_weights[i] = tau * source_weights[i] + (1 - tau) * target_weights[i]
    target.set_weights(target_weights)

def choose_action(state, epsilon):
    if np.random.rand() < epsilon:
        return np.random.randint(action_size)
    q_vals = q_network(np.expand_dims(np.expand_dims(state, 0), 0)).numpy()[0, 0]
    return int(np.argmax(q_vals))

# ============================================================
# 🧾 Logging
# ============================================================
with open(LOG_CSV, "w", newline="") as f:
    csv.writer(f).writerow(["Episode", "TotalReward", "Loss", "Duration_s", "Epsilon"])

# ============================================================
# 🚀 Training Loop
# ============================================================
epsilon = CONFIG["EPS_START"]
print(f"🚀 Starting R2D2 training, logs -> {LOG_CSV}")

for ep in range(1, CONFIG["EPISODES"] + 1):
    start_time = time.time()
    state, _ = env.reset()
    state = state.astype(np.float32)
    done, ep_reward, loss_value = False, 0.0, 0.0
    steps = 0

    while not done and steps < CONFIG["MAX_STEPS"]:
        action = choose_action(state, epsilon)
        next_state, reward, terminated, truncated, _ = env.step(action)
        done = terminated or truncated
        replay_buffer.push(state, action, reward, next_state.astype(np.float32), done)
        ep_reward += reward
        state = next_state.astype(np.float32)
        steps += 1

        # Start learning only after warm-up
        if len(replay_buffer) > 1000:
            states_b, actions_b, rewards_b, next_states_b, dones_b = replay_buffer.sample()
            next_q = target_network(next_states_b).numpy()
            max_next_q = np.max(next_q[:, -1, :], axis=1)
            target = rewards_b[:, -1] + CONFIG["GAMMA"] * max_next_q * (1 - dones_b[:, -1])

            with tf.GradientTape() as tape:
                q_pred = q_network(states_b)
                q_pred_last = tf.reduce_sum(
                    tf.one_hot(actions_b[:, -1], action_size) * q_pred[:, -1, :], axis=1
                )
                loss = loss_fn(target, q_pred_last)

            grads = tape.gradient(loss, q_network.trainable_variables)
            grads, _ = tf.clip_by_global_norm(grads, CONFIG["GRAD_CLIP"])
            optimizer.apply_gradients(zip(grads, q_network.trainable_variables))
            loss_value = float(loss.numpy())

    epsilon = max(CONFIG["EPS_END"], epsilon * CONFIG["EPS_DECAY"])
    duration = time.time() - start_time
    soft_update(target_network, q_network)

    with open(LOG_CSV, "a", newline="") as f:
        csv.writer(f).writerow([ep, ep_reward, loss_value, round(duration, 2), round(epsilon, 3)])

    print(f"🏁 Ep {ep:4d} | R: {ep_reward:8.2f} | L: {loss_value:.4f} | eps: {epsilon:.3f} | t={duration:.2f}s")

    if ep % CONFIG["SAVE_EVERY"] == 0:
        q_network.save(os.path.join(RUN_DIR, f"qnet_ep{ep}.keras"))
        print(f"💾 Saved model at episode {ep}")

env.close()
print("✅ Training finished. Logs:", LOG_CSV)
