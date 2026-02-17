"""
run_all.py
One-command pipeline: runs all 4 steps in order, then launches demo.
Usage: python run_all.py
"""

import subprocess
import sys
import time


def run_step(step_name: str, script: str, extra_args: list = None):
    args = [sys.executable, script] + (extra_args or [])
    print(f"\n{'='*60}")
    print(f"  STEP: {step_name}")
    print(f"  Running: {' '.join(args)}")
    print(f"{'='*60}")
    start = time.time()
    result = subprocess.run(args, check=True)
    elapsed = time.time() - start
    print(f"\n  [OK] {step_name} completed in {elapsed:.1f}s")
    return result


if __name__ == "__main__":
    print("\n" + "=" * 60)
    print("  HYBRID RECOMMENDATION SYSTEM — FULL PIPELINE")
    print("=" * 60)

    run_step("1. Generate Synthetic Data",    "01_generate_data.py")
    run_step("2. Feature Engineering",        "02_preprocessing.py")
    run_step("3. Market Basket Analysis",     "03_market_basket.py")
    run_step("4. KMeans Clustering",          "04_train_kmeans.py", ["--n_clusters", "4", "--plot"])
    run_step("5. Recommendation Demo",        "05_recommender.py")

    print("\n" + "=" * 60)
    print("  ALL STEPS COMPLETE!")
    print("  Run interactive mode: python 05_recommender.py --interactive")
    print("=" * 60)
