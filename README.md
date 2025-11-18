Reinforcement Learning on Airport Queuing Environment

This repository contains the complete codebase, logs, and trained models for experimenting with multiple Reinforcement Learning (RL) algorithms on a custom-designed Airport Environment.
The project evaluates four RL algorithms — PPO, R2D2, A2C, and DynaQ — across multiple random seeds to analyze consistency, stability, and performance.

📁 Repository Structure
1. Main Components

airport_env.py
Contains the full implementation of the custom Airport Environment, used as the simulation testbed for all experiments.

Training & Testing Scripts
For each algorithm, the repository includes separate Python files for:

Training (e.g., train_ppo.py, train_a2c.py, etc.)
Testing/Evaluation (e.g., test_ppo.py, etc.)

2. CSV Test Logs

Four CSV files starting with the prefix tests_*.csv store:
Performance logs
Episode rewards
Metrics for each algorithm during testing
Each file corresponds to one of the four RL algorithms.

3. Visualization Tools

The repo contains three dedicated visualization scripts:
Reward Visualization Script – Plots reward curves using Matplotlib
Plotly-based Interactive Reward Visualization – For zoomable, interactive plots
Loss Evolution Visualization – Displays how training loss evolves over time

4. Logs Directory

logs/ contains:

Training logs for each algorithm
Organized by algorithm name and SEED values
Useful for reproducing results or debugging training behavior

5. Runs Directory

runs/ contains:

Trained model weights for all experiments
Saved using algorithm-specific conventions
Can be directly loaded to reproduce test results

🎯 Trained Models Overview

A total of 20 models were trained:

4 Algorithms: PPO R2D2 A2C DynaQ
5 Random Seeds: 10, 20, 30, 40, 50

Each algorithm was trained on all 5 seed values → 4 × 5 = 20 trained models.

This allows comparative analysis of: Stability across seeds

Algorithmic variance

Generalization across runs
