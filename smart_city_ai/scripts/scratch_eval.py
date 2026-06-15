import os
import sys
from pathlib import Path

repo_root = Path(os.path.dirname(os.path.dirname(__file__)))
os.chdir(repo_root)
sys.path.append(str(repo_root))

from environment.sumo_environment import SumoEnvironment
from rl_agents.dqn_agent import DQNAgent
from rl_agents.ppo_agent import PPOAgent

config_file = repo_root / "sumo_network" / "config.sumocfg"
env = SumoEnvironment(str(config_file), use_gui=False, num_seconds=3600)

for agent_name, ep in [("dqn", 150), ("ppo", 200)]:
    model_path = repo_root / "models" / agent_name / f"{agent_name}_model_ep_{ep}.pth"
    print(f"\n==================== {agent_name.upper()} EP {ep} ====================")
    if not model_path.exists():
        print("Missing")
        continue
    
    if agent_name == "dqn":
        agent = DQNAgent(env.state_dim, env.action_dim)
    else:
        agent = PPOAgent(env.state_dim, env.action_dim)
    
    agent.load(str(model_path))
    
    state = env.reset(seed=9999)
    done = False
    
    while not done:
        if agent_name == "ppo":
            action, _ = agent.act(state)
        else:
            action = agent.act(state)
        state, reward, done, _ = env.step(action)
        
    metrics = env.get_episode_metrics()
    print("Metrics:", metrics)
    if "components" in metrics:
        comps = metrics["components"]
        for k, v in sorted(comps.items(), key=lambda x: abs(x[1]), reverse=True):
            print(f"  {k}: {v}")

env.close()
