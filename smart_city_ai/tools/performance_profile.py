#!/usr/bin/env python3
"""Minimal performance profiler for the hybrid RL stack."""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

repo_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(repo_root))

from integration.hybrid_env import HybridSumoEnv
from rl_agents.dqn_agent import DQNAgent
from rl_agents.ddqn_agent import DDQNAgent
from rl_agents.ppo_agent import PPOAgent
from utils.checkpoint_registry import resolve_active_checkpoint
from utils.project_paths import resolve_first_existing


def build_agent(agent_type: str, state_dim: int, action_dim: int):
    if agent_type == "dqn":
        return DQNAgent(state_dim, action_dim)
    if agent_type == "ddqn":
        return DDQNAgent(state_dim, action_dim)
    if agent_type == "ppo":
        return PPOAgent(state_dim, action_dim)
    raise ValueError(f"Unsupported agent type: {agent_type}")


def profile_agent(agent_type: str, steps: int, seed: int):
    config_file = repo_root / "sumo_network" / "config.sumocfg"
    pinn_model_path = resolve_first_existing(
        repo_root,
        [
            Path("pinn_models") / "trained_pinn.pth",
            Path("..") / "results" / "checkpoint_backups" / "20260530_203534" / "trained_pinn.pth",
        ],
    )
    if pinn_model_path is None:
        raise FileNotFoundError("No compatible PINN checkpoint found for profiling")

    env = HybridSumoEnv(str(config_file), str(pinn_model_path), use_gui=False, num_seconds=steps)
    agent = build_agent(agent_type, env.state_dim, env.action_dim)

    agent_path = Path(resolve_active_checkpoint(agent_type))
    load_started = time.perf_counter()
    agent.load(str(agent_path))
    load_seconds = time.perf_counter() - load_started

    action_latencies = []
    env_step_latencies = []
    try:
        state = env.reset(seed=seed)
        for _ in range(steps):
            action_started = time.perf_counter()
            if agent_type == "ppo":
                action, _ = agent.act(state, eval_mode=True)
            else:
                action = agent.act(state, eval_mode=True)
            action_latencies.append(time.perf_counter() - action_started)

            step_started = time.perf_counter()
            state, _, done, _ = env.step(action)
            env_step_latencies.append(time.perf_counter() - step_started)
            if done:
                break
        metrics = env.get_episode_metrics()
    finally:
        if hasattr(env, "close"):
            env.close()

    return {
        "agent": agent_type,
        "checkpoint": str(agent_path),
        "pinn_checkpoint": str(pinn_model_path),
        "load_seconds": load_seconds,
        "mean_action_seconds": sum(action_latencies) / len(action_latencies) if action_latencies else 0.0,
        "mean_env_step_seconds": sum(env_step_latencies) / len(env_step_latencies) if env_step_latencies else 0.0,
        "max_env_step_seconds": max(env_step_latencies) if env_step_latencies else 0.0,
        "steps_executed": len(env_step_latencies),
        "reward": float(metrics.get("reward", 0.0)),
        "wait": float(metrics.get("wait", 0.0)),
        "queue": float(metrics.get("queue", 0.0)),
        "throughput": float(metrics.get("throughput", 0.0)),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Profile hybrid agent and environment latency")
    parser.add_argument("--steps", type=int, default=300)
    parser.add_argument("--seed", type=int, default=777)
    parser.add_argument("--out", default=str(repo_root / "results" / "performance" / "performance_profile.json"))
    args = parser.parse_args()

    os.makedirs(Path(args.out).parent, exist_ok=True)
    payload = {agent: profile_agent(agent, args.steps, args.seed) for agent in ("dqn", "ddqn", "ppo")}
    with open(args.out, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2)
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
