# Reinforcement Learning on Airport Queuing Environment

This repository contains the complete codebase, logs, and trained models for experimenting with multiple Reinforcement Learning (RL) algorithms on a custom-designed Airport Environment.

The project evaluates four RL algorithms — PPO, R2D2, A2C, and Dyna-Q — across multiple random seeds to analyze consistency, stability, and performance.

---

## 📁 Repository Structure

### **1. Core Components**

**airport_env.py**
Contains the full implementation of the custom Airport Environment, used as the simulation testbed for all experiments.

**Training & Testing Scripts**
Each algorithm includes dedicated scripts for:

* Training (e.g., `train_ppo.py`, `train_a2c.py`)
* Testing/Evaluation (e.g., `test_ppo.py`)

---

### **2. CSV Test Logs**

CSV files (`tests_*.csv`) store:

* Episode-wise rewards
* Evaluation metrics
* Algorithm-specific performance logs

Each file corresponds to one RL algorithm.

---

### **3. Visualization Tools**

The repository includes scripts for:

* Reward visualization (Matplotlib)
* Interactive reward plots (Plotly)
* Loss evolution analysis

---

### **4. Logs Directory (`logs/`)**

Contains:

* Training logs for all algorithms
* Organized by algorithm and SEED values
* Useful for debugging and reproducibility

---

### **5. Runs Directory (`runs/`)**

Contains:

* Trained model weights
* Saved checkpoints for all experiments
* Can be reused for evaluation

---

## 🎯 Trained Models Overview

A total of **20 models** were trained:

* **Algorithms:** PPO, R2D2, A2C, Dyna-Q
* **Seeds:** 10, 20, 30, 40, 50

Each algorithm is trained across all seeds → **4 × 5 = 20 models**

This setup enables:

* Stability analysis across seeds
* Algorithmic comparison
* Generalization evaluation

---

## 📊 Results Summary

### 🔹 Phase 1 (Baseline Environment)

* **A2C**

  * Achieved the highest training convergence (~15–25% above baseline)
  * Initially strong test performance but degraded significantly over longer evaluations
  * Indicates convergence to **local optima** and instability in long horizons

* **Dyna-Q**

  * Most **stable and consistent** across seeds
  * Performance remained **below baseline (~40% lower)**
  * Demonstrates robustness but limited optimality

* **PPO**

  * Highly unstable across seeds
  * Frequently diverged to **large negative rewards**
  * Sensitive to clipping and initial policy conditions

* **R2D2**

  * Stable training behavior with low loss magnitude
  * Poor generalization during evaluation
  * Suggests **over-reliance on memorization (recurrent memory effects)**

---

### 🔹 Phase 2 (Focused Experiments)

* **A2C (Delay Removal)**

  * Removing service switch delay improved **training stability**
  * However, resulted in **lower overall reward**
  * Demonstrates sensitivity to **delayed reward signals (critic misestimation)**

* **Dyna-Q (Planning Steps)**

  * Varying planning steps (0–20) showed **minimal impact on performance**
  * Both reward and loss remained nearly unchanged
  * Suggests **planning plays a limited role in complex stochastic queue systems**

* **Unconstrained Environment (A2C & Dyna-Q)**

  * Models trained without constraints performed:

    * **Better in constrained environments**
    * **Worse in unconstrained environments**
  * Dyna-Q showed **severe degradation (negative rewards)** without limits
  * Highlights **non-trivial generalization behavior in RL policies**

---

## 🔬 Phase 2 Experiments

The `Phase2_experiments/` directory contains extended studies focused on understanding specific behavioral aspects of RL algorithms in the queueing environment.

### 📂 Directory Overview

```
Phase2_experiments/
│── No_Upper_Limit/
│── planning_in_dynaQ/
│── service_delay_removal_a2c/
```

---

### **1. Planning in Dyna-Q**

**Path:** `planning_in_dynaQ/`
**Focus:** Impact of planning steps (k)

Contents:

* `dynaQ_planning_step_tests.csv` → Evaluation results
* `visualize_dynaQ_reward.py` → Reward trends
* `visualize_loss.py` → Loss evolution
* `runs/` → Model checkpoints

---

### **2. Service Delay Removal (A2C)**

**Path:** `service_delay_removal_a2c/`
**Focus:** Effect of delayed reward realization

Contents:

* `a2c_env_comparison_tests.csv` → Performance comparison
* `baseline_without_delay_results_summary.csv` → Baseline stats
* `visualize_a2c_reward.py` → Reward plots
* `visualize_loss.py` → Loss comparison
* `runs/` → Trained models

---

### **3. Training Without Upper Resource Limits**

**Path:** `No_Upper_Limit/`

Focus:

* Behavior in unconstrained environments

Removes:

* Service capacity limits
* Resource allocation constraints

Purpose:

* Explore extreme policy behaviors
* Analyze cross-environment generalization

---

## 📌 Key Insights

* A2C is highly sensitive to **delayed reward structures**
* Dyna-Q prioritizes **stability over optimality**
* Planning does **not significantly enhance performance** in complex queue systems
* RL agents may **generalize better to constrained environments** than the ones they are trained in
* Removing constraints can **reduce robustness rather than improve performance**

---

## 🚀 Reproducibility

All experiments are reproducible using:

* Saved configurations
* Logged results
* Stored model weights

---

## 🔗 Project Repository

GitHub Link:
https://github.com/ojas-khamele/Reinforcement_Learning_for_MultiStage_Queue

---
