#!/usr/bin/env python3
"""
Comprehensive RL Benchmark and Evaluation

Evaluates all RL algorithms:
- DQN
- DDQN
- PPO
- Fixed-Time Baseline

Generates:
- Detailed comparison statistics
- Performance reports
- Visualization data
"""
import os
import sys
import json
import csv
import numpy as np
from pathlib import Path
from statistics import mean, stdev, StatisticsError
from collections import defaultdict

import torch

repo_root = Path(__file__).parent.parent
sys.path.insert(0, str(repo_root))

from integration.hybrid_env import HybridSumoEnv
from rl_agents.dqn_agent import DQNAgent
from rl_agents.ddqn_agent import DDQNAgent
from rl_agents.ppo_agent import PPOAgent
from utils.checkpoint_registry import resolve_active_checkpoint
from utils.reliability import save_json


class RLBenchmark:
    """Comprehensive RL benchmark suite."""
    
    def __init__(self, eval_duration=600, seeds=None, output_dir=None):
        self.eval_duration = eval_duration
        self.seeds = seeds or [101, 202, 303, 404, 505]
        self.output_dir = Path(output_dir or repo_root / "results/rl_benchmark")
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        self.config_file = repo_root / "sumo_network/config.sumocfg"
        self.pinn_path = repo_root / "pinn_models/trained_pinn.pth"
        
        self.results = defaultdict(lambda: defaultdict(dict))
    
    def _load_agent(self, agent_type):
        """Load trained agent checkpoint."""
        try:
            model_path = resolve_active_checkpoint(agent_type)
        except Exception:
            results_dir = repo_root / "results/hybrid_results"
            candidates = [
                results_dir / f"hybrid_{agent_type}_model_best.pth",
                results_dir / f"hybrid_{agent_type}_model_final.pth",
            ]
            model_path = next((p for p in candidates if p.exists()), None)
        
        if model_path is None:
            return None, None
        
        # Create environment to get state/action dims
        use_gui = bool(getattr(self, '_gui', False))
        env = HybridSumoEnv(str(self.config_file), str(self.pinn_path), use_gui=use_gui, num_seconds=self.eval_duration)

        state_dim = env.state_dim
        action_dim = env.action_dim
        env.close()
        
        # Load agent
        if agent_type == 'dqn':
            agent = DQNAgent(state_dim, action_dim)
        elif agent_type == 'ddqn':
            agent = DDQNAgent(state_dim, action_dim)
        elif agent_type == 'ppo':
            agent = PPOAgent(state_dim, action_dim)
        else:
            return None, None
        
        agent.load(str(model_path))
        return agent, env
    
    def _eval_fixed_time(self, seed, phase_durations=None):
        """Evaluate fixed-time baseline."""
        if phase_durations is None:
            phase_durations = [30, 15, 30, 15]  # Default phases
        
        use_gui = bool(getattr(self, '_gui', False))
        env = HybridSumoEnv(str(self.config_file), str(self.pinn_path), use_gui=use_gui, num_seconds=self.eval_duration)
        env.reset(seed=seed)

        
        done = False
        phase_idx = 0
        
        while not done and env.current_step < self.eval_duration:
            duration = max(
                float(phase_durations[phase_idx % len(phase_durations)]),
                getattr(env, 'min_green', 1)
            )
            remaining = min(duration, max(0, self.eval_duration - env.current_step))
            calls = int(np.ceil(remaining / max(1, env.action_repeat)))
            
            for _ in range(calls):
                _, _, done, _ = env.step(0)
                if done or env.current_step >= self.eval_duration:
                    break
            
            phase_idx += 1
        
        metrics = env.get_episode_metrics()
        env.close()
        return metrics
    
    def _eval_agent(self, agent_type, agent, seed):
        """Evaluate RL agent."""
        use_gui = bool(getattr(self, '_gui', False))
        env = HybridSumoEnv(str(self.config_file), str(self.pinn_path), use_gui=use_gui, num_seconds=self.eval_duration)
        state = env.reset(seed=seed)

        
        done = False
        while not done and env.current_step < self.eval_duration:
            if agent_type == 'ppo':
                action, _ = agent.act(state, eval_mode=True)
            else:
                action = agent.act(state, eval_mode=True)
            
            state, _, done, _ = env.step(action)
        
        metrics = env.get_episode_metrics()
        env.close()
        return metrics
    
    def evaluate_all(self):
        """Run complete benchmark."""
        print(f"\n{'='*80}")
        print(f"RL BENCHMARK EVALUATION ({len(self.seeds)} seeds)")
        print(f"{'='*80}\n")
        
        # Fixed-Time Baseline
        print("Evaluating Fixed-Time Baseline...")
        for seed in self.seeds:
            metrics = self._eval_fixed_time(seed)
            self.results['fixed_time'][seed] = metrics
            print(f"  Seed {seed}: Wait={metrics['wait']:.2f}s, Queue={metrics['queue']:.2f}, "
                  f"Throughput={metrics['throughput']:.0f}")
        
        # RL Agents
        for agent_type in ['dqn', 'ddqn', 'ppo']:
            print(f"\nEvaluating {agent_type.upper()}...")
            agent, _ = self._load_agent(agent_type)
            
            if agent is None:
                print(f"  ERROR: Could not load {agent_type} model")
                continue
            
            for seed in self.seeds:
                try:
                    metrics = self._eval_agent(agent_type, agent, seed)
                    self.results[agent_type][seed] = metrics
                    print(f"  Seed {seed}: Wait={metrics['wait']:.2f}s, Queue={metrics['queue']:.2f}, "
                          f"Throughput={metrics['throughput']:.0f}")
                except Exception as e:
                    print(f"  Seed {seed}: ERROR - {e}")
                    self.results[agent_type][seed] = None
    
    def compute_statistics(self):
        """Compute aggregate statistics."""
        stats = {}
        
        for agent_type in ['fixed_time', 'dqn', 'ddqn', 'ppo']:
            agent_results = self.results[agent_type]
            valid_results = [m for m in agent_results.values() if m is not None]
            
            if not valid_results:
                continue
            
            metrics_keys = ['wait', 'queue', 'avg_speed', 'throughput', 'switches']
            stats[agent_type] = {}
            
            for metric in metrics_keys:
                values = [float(m.get(metric, 0.0)) for m in valid_results]
                stats[agent_type][metric] = {
                    'mean': float(mean(values)) if values else 0.0,
                    'std': float(stdev(values)) if len(values) > 1 else 0.0,
                    'min': float(min(values)) if values else 0.0,
                    'max': float(max(values)) if values else 0.0,
                    'ci_95': self._compute_ci_95(values),
                }
        
        return stats
    
    def _compute_ci_95(self, values):
        """Compute 95% confidence interval."""
        if len(values) < 2:
            return [values[0] if values else 0.0, values[0] if values else 0.0]
        
        mu = mean(values)
        s = stdev(values)
        n = len(values)
        t_values = {1: 12.706, 2: 4.303, 3: 3.182, 4: 2.776, 5: 2.571, 6: 2.447, 7: 2.365, 8: 2.306}
        t = t_values.get(n - 1, 1.96)
        se = s / np.sqrt(n)
        return [mu - t * se, mu + t * se]
    
    def generate_report(self, stats):
        """Generate comprehensive report."""
        report = {
            'title': 'RL Traffic Signal Control Benchmark',
            'eval_duration': self.eval_duration,
            'num_seeds': len(self.seeds),
            'seeds': self.seeds,
            'timestamp': str(Path.cwd()),
            'statistics': stats,
            'per_seed_results': dict(self.results),
        }
        
        # Generate ranking
        if 'wait' in stats.get('fixed_time', {}):
            ranking = sorted(
                [(agent, stats[agent]['wait']['mean']) for agent in stats.keys()],
                key=lambda x: x[1]
            )
            report['ranking_by_wait'] = ranking
        
        # Compute improvements over baseline
        if 'wait' in stats.get('fixed_time', {}) and 'wait' in stats.get('dqn', {}):
            baseline_wait = stats['fixed_time']['wait']['mean']
            improvements = {}
            
            for agent in ['dqn', 'ddqn', 'ppo']:
                if agent in stats and 'wait' in stats[agent]:
                    agent_wait = stats[agent]['wait']['mean']
                    improvement = (baseline_wait - agent_wait) / baseline_wait * 100.0
                    improvements[agent] = improvement
            
            report['improvements_vs_baseline'] = improvements
        
        return report
    
    def export_csv(self, stats):
        """Export results as CSV."""
        csv_path = self.output_dir / 'benchmark_results.csv'
        
        with open(csv_path, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(['Agent', 'Metric', 'Mean', 'Std', 'Min', 'Max', 'CI_Low', 'CI_High'])
            
            for agent in ['fixed_time', 'dqn', 'ddqn', 'ppo']:
                if agent not in stats:
                    continue
                
                for metric in ['wait', 'queue', 'avg_speed', 'throughput', 'switches']:
                    if metric not in stats[agent]:
                        continue
                    
                    s = stats[agent][metric]
                    writer.writerow([
                        agent,
                        metric,
                        f"{s['mean']:.4f}",
                        f"{s['std']:.4f}",
                        f"{s['min']:.4f}",
                        f"{s['max']:.4f}",
                        f"{s['ci_95'][0]:.4f}",
                        f"{s['ci_95'][1]:.4f}",
                    ])
        
        print(f"\n✓ Exported CSV to {csv_path}")
    
    def run_complete_benchmark(self):
        """Run complete benchmark and generate all reports."""
        self.evaluate_all()
        stats = self.compute_statistics()
        report = self.generate_report(stats)
        self.export_csv(stats)
        
        # Save JSON report
        report_path = self.output_dir / 'benchmark_report.json'
        save_json(report_path, report)
        print(f"✓ Saved report to {report_path}")
        
        # Print summary
        print(f"\n{'='*80}")
        print("BENCHMARK SUMMARY")
        print(f"{'='*80}\n")
        
        if 'statistics' in report:
            for agent in ['fixed_time', 'dqn', 'ddqn', 'ppo']:
                if agent in report['statistics']:
                    s = report['statistics'][agent]
                    if 'wait' in s:
                        print(f"{agent.upper():12} | "
                              f"Wait: {s['wait']['mean']:6.2f}±{s['wait']['std']:5.2f}s | "
                              f"Queue: {s['queue']['mean']:5.2f}±{s['queue']['std']:4.2f} | "
                              f"Throughput: {s['throughput']['mean']:6.0f}±{s['throughput']['std']:5.0f}")
        
        if 'ranking_by_wait' in report:
            print(f"\nRanking by Wait Time (lower is better):")
            for i, (agent, wait) in enumerate(report['ranking_by_wait'], 1):
                print(f"  {i}. {agent:15} {wait:6.2f}s")
        
        if 'improvements_vs_baseline' in report:
            print(f"\nImprovement vs Fixed-Time Baseline:")
            for agent, improvement in report['improvements_vs_baseline'].items():
                symbol = "↓" if improvement > 0 else "↑"
                print(f"  {agent.upper():8} {symbol} {abs(improvement):6.2f}%")
        
        print(f"\n{'='*80}\n")
        
        return report


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--duration', type=int, default=600)
    parser.add_argument('--seeds', nargs='+', type=int, default=[101, 202, 303, 404, 505])
    parser.add_argument('--output', default=None)
    parser.add_argument('--gui', action='store_true', default=False, help='Launch SUMO-GUI instead of SUMO')
    args = parser.parse_args()

    
    benchmark = RLBenchmark(eval_duration=args.duration, seeds=args.seeds, output_dir=args.output)
    benchmark.run_complete_benchmark()
