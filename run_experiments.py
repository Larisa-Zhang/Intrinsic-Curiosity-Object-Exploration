# run_experiments.py
# Offline launcher — trains all experiments in parallel
# Usage: python run_experiments.py

from train_icm_rl import run_all_experiments

if __name__ == "__main__":
    run_all_experiments(seed=1)