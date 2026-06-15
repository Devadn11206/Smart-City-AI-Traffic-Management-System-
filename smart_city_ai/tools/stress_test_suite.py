#!/usr/bin/env python3
"""Short stress suite for the hybrid RL stack.

Runs a small number of seeded hybrid episodes for each agent using the
canonical checkpoints and records a compact JSON summary.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

repo_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(repo_root))

from integration.hybrid_env import HybridSumoEnv
from rl_agents.dqn_agent import DQNAgent
from rl_agents.ddqn_agent import DDQNAgent
from rl_agents.ppo_agent import PPOAgent
from utils.project_paths import resolve_first_existing
from utils.checkpoint_registry import resolve_active_checkpoint


def build_agent(agent_type: str, state_dim: int, action_dim: int):
    if agent_type == "dqn":
        return DQNAgent(state_dim, action_dim)
    if agent_type == "ddqn":
        return DDQNAgent(state_dim, action_dim)
    if agent_type == "ppo":
        return PPOAgent(state_dim, action_dim)
    raise ValueError(f"Unsupported agent type: {agent_type}")


def run_stress_test(agent_type: str, episodes: int, duration: int, seed: int):
    config_file = repo_root / "sumo_network" / "config.sumocfg"
    pinn_model_path = resolve_first_existing(
        repo_root,
        [
            Path("pinn_models") / "trained_pinn.pth",
            Path("..") / "results" / "checkpoint_backups" / "20260530_203534" / "trained_pinn.pth",
        ],
    )
    if pinn_model_path is None:
        raise FileNotFoundError("No compatible PINN checkpoint found for stress testing")

    agent_path = Path(resolve_active_checkpoint(agent_type))
    env = HybridSumoEnv(str(config_file), str(pinn_model_path), use_gui=False, num_seconds=duration)
    agent = build_agent(agent_type, env.state_dim, env.action_dim)
    agent.load(str(agent_path))

    episode_results = []
    try:
        for episode_index in range(episodes):
            episode_seed = seed + episode_index
            state = env.reset(seed=episode_seed)
            done = False
            steps = 0
            while not done and env.current_step < duration:
                if agent_type == "ppo":
                    action, _ = agent.act(state, eval_mode=True)
                else:
                    action = agent.act(state, eval_mode=True)
                state, reward, done, _ = env.step(action)
                steps += 1
            metrics = env.get_episode_metrics()
            episode_results.append(
                {
                    "episode": episode_index,
                    "seed": episode_seed,
                    "steps": steps,
                    "reward": float(metrics.get("reward", 0.0)),
                    "wait": float(metrics.get("wait", 0.0)),
                    "queue": float(metrics.get("queue", 0.0)),
                    "throughput": float(metrics.get("throughput", 0.0)),
                }
            )
    finally:
        if hasattr(env, "close"):
            env.close()

    failures = [item for item in episode_results if item["steps"] <= 0]
    return {
        "agent": agent_type,
        "checkpoint": str(agent_path),
        "pinn_checkpoint": str(pinn_model_path),
        "episodes": episodes,
        "duration": duration,
        "failures": len(failures),
        "results": episode_results,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Run a compact hybrid stress test suite")
    parser.add_argument("--episodes", type=int, default=3)
    parser.add_argument("--duration", type=int, default=1200)
    parser.add_argument("--seed", type=int, default=9000)
    parser.add_argument("--out", default=str(repo_root / "results" / "stress_tests" / "stress_test_summary.json"))
    args = parser.parse_args()

    os.makedirs(Path(args.out).parent, exist_ok=True)
    summary = {agent: run_stress_test(agent, args.episodes, args.duration, args.seed) for agent in ("dqn", "ddqn", "ppo")}
    payload = {"summary": summary}
    with open(args.out, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2)
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
