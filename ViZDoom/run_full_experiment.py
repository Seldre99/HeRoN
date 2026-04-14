#!/usr/bin/env python3
"""
HERON Full Experiment Runner.
"""

import argparse
import json
import os
import subprocess
import sys
from collections import defaultdict
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd


# Configuration
SCENARIOS = ['basic', 'defend_the_center', 'deadly_corridor']
STRATEGIES = ['initial', 'final_01', 'final_02', 'random']
NUM_RUNS = int(os.getenv('NUM_RUNS', '2'))
EPISODES_PER_RUN = int(os.getenv('EPISODES_PER_RUN', '500'))
TIMEOUT_SECONDS = int(os.getenv('TIMEOUT_SECONDS', '7200'))


def run_training(scenario: str, strategy: str, run_id: int, output_dir: str) -> dict:
    """Execute a single training run and return results."""
    print(f"\n{'='*60}")
    print(f"SCENARIO: {scenario} | STRATEGY: {strategy} | RUN: {run_id + 1}/{NUM_RUNS}")
    print(f"{'='*60}")

    cmd = [
        sys.executable,
        "train_DQN.py",
        "--scenario", scenario,
        "--episodes", str(EPISODES_PER_RUN),
        "--strategy", strategy,
        "--no-reviewer",
        "--quiet"
    ]

    run_output_dir = os.path.join(output_dir, f"{scenario}_{strategy}_run{run_id + 1}")
    env = os.environ.copy()
    env['TRAINING_OUTPUT_DIR'] = run_output_dir

    try:
        result = subprocess.run(
            cmd,
            cwd=os.path.dirname(os.path.abspath(__file__)),
            capture_output=True,
            text=True,
            timeout=TIMEOUT_SECONDS
        )

        training_results_dir = "training_results"
        pattern = f"{scenario}_{strategy}"
        matching_dirs = []

        if os.path.exists(training_results_dir):
            for d in os.listdir(training_results_dir):
                if d.startswith(pattern) and os.path.isdir(os.path.join(training_results_dir, d)):
                    full_path = os.path.join(training_results_dir, d)
                    matching_dirs.append((full_path, os.path.getmtime(full_path)))

        if matching_dirs:
            matching_dirs.sort(key=lambda x: x[1], reverse=True)
            latest_dir = matching_dirs[0][0]

            stats_file = os.path.join(latest_dir, "stats.json")
            if os.path.exists(stats_file):
                with open(stats_file, 'r') as f:
                    stats = json.load(f)

                new_dir = os.path.join(output_dir, f"{scenario}_{strategy}_run{run_id + 1}")
                if os.path.exists(new_dir):
                    import shutil
                    shutil.rmtree(new_dir)
                os.rename(latest_dir, new_dir)

                return {
                    'scenario': scenario,
                    'strategy': strategy,
                    'run': run_id + 1,
                    'rewards': stats.get('rewards', []),
                    'wins': stats.get('wins', []),
                    'kills': stats.get('kills', []),
                    'helper_suggestions': stats.get('helper_suggestions', 0),
                    'success': True,
                    'output_dir': new_dir
                }

        return {
            'scenario': scenario,
            'strategy': strategy,
            'run': run_id + 1,
            'rewards': [],
            'wins': [],
            'kills': [],
            'success': False,
            'error': 'Stats file not found'
        }

    except subprocess.TimeoutExpired:
        return {
            'scenario': scenario,
            'strategy': strategy,
            'run': run_id + 1,
            'success': False,
            'error': 'Timeout'
        }
    except Exception as e:
        return {
            'scenario': scenario,
            'strategy': strategy,
            'run': run_id + 1,
            'success': False,
            'error': str(e)
        }


def calculate_cv(values: list) -> float:
    """Calculate Coefficient of Variation (CV = std/mean)."""
    if not values or len(values) < 2:
        return float('nan')

    values = np.array(values)
    mean = np.mean(values)
    if mean == 0:
        return float('nan')

    return np.std(values) / abs(mean)


def aggregate_results(all_results: list) -> pd.DataFrame:
    """Aggregate results from all runs and calculate statistics."""
    aggregated = []
    grouped = defaultdict(list)

    for result in all_results:
        if result.get('success', False):
            key = (result['scenario'], result['strategy'])
            grouped[key].append(result)

    for (scenario, strategy), runs in grouped.items():
        all_rewards = []
        all_wins = []
        all_kills = []

        for run in runs:
            rewards = run.get('rewards', [])
            if rewards:
                final_rewards = rewards[-20:] if len(rewards) >= 20 else rewards
                all_rewards.append(np.mean(final_rewards))

            wins = run.get('wins', [])
            if wins:
                all_wins.append(np.mean(wins))

            kills = run.get('kills', [])
            if kills:
                all_kills.append(np.sum(kills))

        row = {
            'scenario': scenario,
            'strategy': strategy,
            'num_runs': len(runs),
            'reward_mean': np.mean(all_rewards) if all_rewards else 0,
            'reward_std': np.std(all_rewards) if all_rewards else 0,
            'reward_cv': calculate_cv(all_rewards),
            'win_rate_mean': np.mean(all_wins) if all_wins else 0,
            'win_rate_std': np.std(all_wins) if all_wins else 0,
            'win_rate_cv': calculate_cv(all_wins),
            'kills_mean': np.mean(all_kills) if all_kills else 0,
            'kills_std': np.std(all_kills) if all_kills else 0,
            'kills_cv': calculate_cv(all_kills),
            'overall_stability': np.nanmean([
                calculate_cv(all_rewards),
                calculate_cv(all_wins)
            ]) if all_rewards and all_wins else float('nan')
        }

        aggregated.append(row)

    return pd.DataFrame(aggregated)


def load_completed_runs(experiment_dir: str) -> set:
    """Load completed runs from existing experiment folder."""
    completed = set()
    results_file = os.path.join(experiment_dir, 'results_raw.json')

    if os.path.exists(results_file):
        try:
            with open(results_file, 'r') as f:
                results = json.load(f)
            for r in results:
                if r.get('success', False):
                    completed.add((r['scenario'], r['strategy'], r['run']))
            print(f"Found {len(completed)} completed runs in {experiment_dir}")
        except Exception:
            pass

    return completed


def main(resume_dir: str = None):
    """Main experiment loop."""
    if resume_dir:
        experiment_dir = resume_dir
        print("=" * 60)
        print("RESUMING EXPERIMENT")
        print("=" * 60)
    else:
        experiment_dir = f"experiment_helper_only_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        print("=" * 60)
        print("HERON EXPERIMENT - DQN + Helper (No Reviewer)")
        print("=" * 60)

    print(f"Output directory: {experiment_dir}")
    print(f"Scenarios: {SCENARIOS}")
    print(f"Strategies: {STRATEGIES}")
    print(f"Runs per combination: {NUM_RUNS}")
    print(f"Episodes per run: {EPISODES_PER_RUN}")
    print(f"Total training runs: {len(SCENARIOS) * len(STRATEGIES) * NUM_RUNS}")
    print("=" * 60)

    completed_runs = load_completed_runs(experiment_dir) if resume_dir else set()
    os.makedirs(experiment_dir, exist_ok=True)

    config = {
        'scenarios': SCENARIOS,
        'strategies': STRATEGIES,
        'num_runs': NUM_RUNS,
        'episodes_per_run': EPISODES_PER_RUN,
        'start_time': datetime.now().isoformat(),
        'mode': 'DQN + Helper (no Reviewer)'
    }

    with open(os.path.join(experiment_dir, 'experiment_config.json'), 'w') as f:
        json.dump(config, f, indent=2)

    all_results = []
    if resume_dir:
        results_file = os.path.join(experiment_dir, 'results_raw.json')
        if os.path.exists(results_file):
            try:
                with open(results_file, 'r') as f:
                    all_results = json.load(f)
                print(f"Loaded {len(all_results)} previous results")
            except Exception:
                pass

    total_runs = len(SCENARIOS) * len(STRATEGIES) * NUM_RUNS
    current_run = 0

    for scenario in SCENARIOS:
        for strategy in STRATEGIES:
            for run_id in range(NUM_RUNS):
                current_run += 1

                if (scenario, strategy, run_id + 1) in completed_runs:
                    print(f"\n[{current_run}/{total_runs}] SKIP (completed): {scenario}/{strategy}/run{run_id + 1}")
                    continue

                print(f"\n[{current_run}/{total_runs}] Starting {scenario}/{strategy}/run{run_id + 1}...")

                result = run_training(scenario, strategy, run_id, experiment_dir)
                all_results.append(result)

                with open(os.path.join(experiment_dir, 'results_raw.json'), 'w') as f:
                    serializable = []
                    for r in all_results:
                        r_copy = r.copy()
                        for k, v in r_copy.items():
                            if isinstance(v, np.ndarray):
                                r_copy[k] = v.tolist()
                        serializable.append(r_copy)
                    json.dump(serializable, f, indent=2)

    print("\n" + "=" * 60)
    print("AGGREGATING RESULTS...")
    print("=" * 60)

    df = aggregate_results(all_results)

    csv_path = os.path.join(experiment_dir, 'experiment_results.csv')
    df.to_csv(csv_path, index=False)
    print(f"\nCSV saved: {csv_path}")

    print("\n" + "=" * 60)
    print("FINAL RESULTS SUMMARY")
    print("=" * 60)

    if not df.empty:
        summary = df[['scenario', 'strategy', 'reward_mean', 'reward_cv',
                      'win_rate_mean', 'win_rate_cv', 'kills_mean', 'overall_stability']]
        summary.columns = ['Scenario', 'Strategy', 'Reward', 'Reward CV',
                          'Win Rate', 'Win Rate CV', 'Kills', 'Stability']
        print(summary.to_string(index=False))

        if not df['overall_stability'].isna().all():
            best_stability = df.loc[df['overall_stability'].idxmin()]
            print(f"\nMost stable: {best_stability['scenario']} + {best_stability['strategy']} (CV={best_stability['overall_stability']:.3f})")

        best_reward = df.loc[df['reward_mean'].idxmax()]
        print(f"Best reward: {best_reward['scenario']} + {best_reward['strategy']} (Reward={best_reward['reward_mean']:.2f})")
    else:
        print("No successful runs to aggregate.")

    config['end_time'] = datetime.now().isoformat()
    with open(os.path.join(experiment_dir, 'experiment_config.json'), 'w') as f:
        json.dump(config, f, indent=2)

    print(f"\nExperiment complete! Results in: {experiment_dir}/")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='HERON Full Experiment Runner')
    parser.add_argument('--resume', type=str, default=None,
                       help='Resume from existing experiment folder')
    args = parser.parse_args()

    main(resume_dir=args.resume)
