import gymnasium as gym
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models, optimizers
import os, json, csv, time
from datetime import datetime
from airport_env import AirportEnv
import random

# ---------------------------
# Reproducibility
# ---------------------------
SEED = 50
np.random.seed(SEED)
tf.random.set_seed(SEED)
random.seed(SEED)

# ============================================================
# Environment Setup
# ============================================================
env = AirportEnv()
env.reset(seed=SEED)
env.action_space.seed(SEED)

state_dim = env.observation_space.shape[0]
action_dim = env.action_space.n

# ============================================================
# PPO Hyperparameters (auto-stable for long training)
# ============================================================
EPISODES = 1000
GAMMA = 0.99
LR_ACTOR = 3e-4
LR_CRITIC = 1e-4
EPSILON_CLIP = 0.18
UPDATE_EPOCHS = 10
BATCH_SIZE = 64
MAX_STEPS = 300
LAMBDA = 0.95
ENTROPY_COEFF = 0.08
REWARD_SCALE = 50.0
GRAD_NORM_CLIP = 0.5
MINIBATCH_MIN = 16
VALUE_CLIP = 0.2
ENTROPY_FLOOR = 0.01     # ✅ ensures exploration never collapses
ENTROPY_BOOST = 0.02     # ✅ small periodic entropy boost
BOOST_INTERVAL = 20      # ✅ every 20 episodes
LR_DECAY = 0.98          # ✅ slow decay for long stability
# ============================================================


# ------------------------------------------------------------
# Run directory and config saving (A2C-style)
# ------------------------------------------------------------
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
run_dir = f"runs/ppo_run_{timestamp}"
os.makedirs(run_dir, exist_ok=True)

CONFIG = {
    "ALGO": "PPO",
    "EPISODES": EPISODES,
    "GAMMA": GAMMA,
    "LR_ACTOR": LR_ACTOR,
    "LR_CRITIC": LR_CRITIC,
    "EPSILON_CLIP": EPSILON_CLIP,
    "UPDATE_EPOCHS": UPDATE_EPOCHS,
    "BATCH_SIZE": BATCH_SIZE,
    "MAX_STEPS": MAX_STEPS,
    "LAMBDA": LAMBDA,
    "ENTROPY_COEFF": ENTROPY_COEFF,
    "REWARD_SCALE": REWARD_SCALE,
    "GRAD_NORM_CLIP": GRAD_NORM_CLIP,
    "MINIBATCH_MIN": MINIBATCH_MIN,
    "VALUE_CLIP": VALUE_CLIP,
    "ENTROPY_FLOOR": ENTROPY_FLOOR,
    "ENTROPY_BOOST": ENTROPY_BOOST,
    "BOOST_INTERVAL": BOOST_INTERVAL,
    "LR_DECAY": LR_DECAY,
}

with open(os.path.join(run_dir, "config.json"), "w") as f:
    json.dump(CONFIG, f, indent=2)

# ------------------------------------------------------------
# Networks
# ------------------------------------------------------------
def build_actor():
    inputs = layers.Input(shape=(state_dim,))
    x = layers.Dense(128, activation='relu')(inputs)
    x = layers.Dense(128, activation='relu')(x)
    outputs = layers.Dense(action_dim, activation='softmax')(x)
    model = models.Model(inputs, outputs)
    model.compile(optimizer=optimizers.Adam(LR_ACTOR))
    return model

def build_critic():
    inputs = layers.Input(shape=(state_dim,))
    x = layers.Dense(128, activation='relu')(inputs)
    x = layers.Dense(128, activation='relu')(x)
    outputs = layers.Dense(1)(x)
    model = models.Model(inputs, outputs)
    model.compile(optimizer=optimizers.Adam(LR_CRITIC))
    return model

actor = build_actor()
critic = build_critic()

# ------------------------------------------------------------
# Logging setup
# ------------------------------------------------------------
log_file = os.path.join(run_dir, "training_log.csv")
with open(log_file, "w", newline="") as f:
    writer = csv.writer(f)
    writer.writerow([
        "Episode", "Reward", "Steps", "Duration(s)",
        "ActorLoss", "CriticLoss", "Entropy",
        "AdvantageMean", "AdvantageStd", "ValueMean"
    ])

# ------------------------------------------------------------
# Helper Functions
# ------------------------------------------------------------
def get_action(state):
    state = np.expand_dims(state, axis=0)
    probs = actor.predict(state, verbose=0)[0]
    probs = np.clip(probs, 1e-8, 1.0)
    probs /= np.sum(probs)
    action = np.random.choice(action_dim, p=probs)
    return action, probs[action], probs

def compute_advantages(rewards, values, next_value, dones):
    advantages, gae = [], 0
    for t in reversed(range(len(rewards))):
        mask = 1.0 - float(dones[t])
        delta = rewards[t] + GAMMA * mask * next_value - values[t]
        gae = delta + GAMMA * LAMBDA * mask * gae
        advantages.insert(0, gae)
        next_value = values[t]
    advantages = np.array(advantages, dtype=np.float32)
    targets = advantages + np.array(values, dtype=np.float32)
    return advantages, targets

# ------------------------------------------------------------
# Training Loop
# ------------------------------------------------------------
for ep in range(1, EPISODES + 1):
    start_time = time.time()
    state, _ = env.reset()
    done = False

    states, actions, rewards, old_probs, dones, values = [], [], [], [], [], []
    total_reward, step = 0, 0

    while not done and step < MAX_STEPS:
        value = critic.predict(np.expand_dims(state, axis=0), verbose=0)[0][0]
        action, prob, _ = get_action(state)
        next_state, reward, terminated, truncated, _ = env.step(action)
        done_flag = bool(terminated or truncated)
        r_scaled = float(reward) / REWARD_SCALE

        states.append(state)
        actions.append(action)
        rewards.append(r_scaled)
        old_probs.append(prob)
        values.append(value)
        dones.append(done_flag)

        state = next_state
        total_reward += reward
        step += 1
        done = done_flag

    next_value = critic.predict(np.expand_dims(state, axis=0), verbose=0)[0][0]
    advantages, targets = compute_advantages(rewards, values, next_value, dones)
    advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

    states = np.array(states, dtype=np.float32)
    actions = np.array(actions, dtype=np.int32)
    old_probs = np.array(old_probs, dtype=np.float32)
    targets = np.array(targets, dtype=np.float32)

    avg_adv, std_adv, avg_val = (
        np.mean(advantages), np.std(advantages), np.mean(values)
    ) if len(advantages) > 0 else (0.0, 0.0, 0.0)

    total_actor_loss, total_critic_loss, total_entropy = 0.0, 0.0, 0.0

    n_samples = len(states)
    if n_samples > 0:
        mb_size = max(MINIBATCH_MIN, BATCH_SIZE)
        for _ in range(UPDATE_EPOCHS):
            idxs = np.random.permutation(n_samples)
            for start_idx in range(0, n_samples, mb_size):
                mb_idx = idxs[start_idx:start_idx+mb_size]
                mb_states = states[mb_idx]
                mb_actions = actions[mb_idx]
                mb_old_probs = old_probs[mb_idx]
                mb_advantages = advantages[mb_idx]
                mb_targets = targets[mb_idx]

                with tf.GradientTape(persistent=True) as tape:
                    probs = actor(mb_states)
                    indices = tf.stack([tf.range(len(mb_actions)), mb_actions], axis=1)
                    action_probs = tf.gather_nd(probs, indices)

                    ratios = action_probs / (mb_old_probs + 1e-10)
                    surr1 = ratios * mb_advantages
                    surr2 = tf.clip_by_value(ratios, 1 - EPSILON_CLIP, 1 + EPSILON_CLIP) * mb_advantages
                    entropy = -tf.reduce_mean(tf.reduce_sum(probs * tf.math.log(probs + 1e-10), axis=1))

                    entropy = tf.maximum(entropy, ENTROPY_FLOOR)

                    actor_loss = -tf.reduce_mean(tf.minimum(surr1, surr2)) - ENTROPY_COEFF * entropy

                    critic_values = critic(mb_states)
                    value_clipped = tf.stop_gradient(critic_values + tf.clip_by_value(
                        critic_values - tf.expand_dims(mb_targets, -1),
                        -VALUE_CLIP, VALUE_CLIP))
                    critic_loss1 = tf.square(mb_targets - tf.squeeze(critic_values))
                    critic_loss2 = tf.square(mb_targets - tf.squeeze(value_clipped))
                    critic_loss = tf.reduce_mean(tf.maximum(critic_loss1, critic_loss2))

                actor_grads = tape.gradient(actor_loss, actor.trainable_variables)
                critic_grads = tape.gradient(critic_loss, critic.trainable_variables)

                actor_grads = [tf.clip_by_norm(g, GRAD_NORM_CLIP) if g is not None else None for g in actor_grads]
                critic_grads = [tf.clip_by_norm(g, GRAD_NORM_CLIP) if g is not None else None for g in critic_grads]

                actor.optimizer.apply_gradients(zip(actor_grads, actor.trainable_variables))
                critic.optimizer.apply_gradients(zip(critic_grads, critic.trainable_variables))
                del tape

                total_actor_loss += float(actor_loss.numpy())
                total_critic_loss += float(critic_loss.numpy())
                total_entropy += float(entropy.numpy())

    denom = UPDATE_EPOCHS * max(1, int(np.ceil(n_samples / max(MINIBATCH_MIN, BATCH_SIZE))))
    avg_actor_loss = total_actor_loss / denom
    avg_critic_loss = total_critic_loss / denom
    avg_entropy = total_entropy / denom
    duration = time.time() - start_time

    with open(log_file, "a", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([
            ep, total_reward, step, round(duration, 2),
            round(avg_actor_loss, 6), round(avg_critic_loss, 6),
            round(avg_entropy, 6), round(avg_adv, 6),
            round(std_adv, 6), round(avg_val, 6)
        ])

    print(f"🏁 Ep {ep:4d} | R: {total_reward:7.2f} | Steps: {step:3d} | "
          f"A_Loss={avg_actor_loss:.4f} | C_Loss={avg_critic_loss:.4f} | "
          f"Entropy={avg_entropy:.4f} | Time={duration:.2f}s")

    if ep % BOOST_INTERVAL == 0:
        ENTROPY_COEFF = min(ENTROPY_COEFF + ENTROPY_BOOST, 0.15)
        actor.optimizer.learning_rate.assign(actor.optimizer.learning_rate * LR_DECAY)
        critic.optimizer.learning_rate.assign(critic.optimizer.learning_rate * LR_DECAY)
        print(f"🔄 Boosting exploration | New ENTROPY_COEFF={ENTROPY_COEFF:.3f}")

    if ep % 10 == 0:
        actor.save(os.path.join(run_dir, f"actor_ep{ep}.keras"))
        critic.save(os.path.join(run_dir, f"critic_ep{ep}.keras"))
        print(f"💾 Saved models at Episode {ep}")

env.close()
print("✅ PPO Training Completed Successfully!")
