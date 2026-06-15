import os
import sys
import argparse
from pathlib import Path

repo_root = Path(__file__).resolve().parent.parent
if str(repo_root) not in sys.path:
    sys.path.insert(0, str(repo_root))

from environment.sumo_environment import SumoEnvironment
from rl_agents.dqn_agent import DQNAgent
from rl_agents.ddqn_agent import DDQNAgent
from rl_agents.ppo_agent import PPOAgent

def main():
    parser = argparse.ArgumentParser(description="Visualize a trained RL agent in SUMO.")
    parser.add_argument('--agent', type=str, default='dqn', choices=['dqn', 'ddqn', 'ppo', 'fixed'], help='Which agent to visualize')
    args = parser.parse_args()

    config_file = str((repo_root / 'sumo_network' / 'config.sumocfg').resolve())
    
    if not os.path.exists(config_file):
        print("Config file not found. Ensure you are running from the smart_city_ai dir, e.g., 'python scripts/visualize_agent.py'")
        return

    print(f"Starting visualization for {args.agent.upper()} agent...")
    # use_gui=True enables the SUMO GUI
    env = SumoEnvironment(config_file, use_gui=True, num_seconds=2000)

    # Initialize agent
    if args.agent == 'dqn':
        agent = DQNAgent(env.state_dim, env.action_dim)
        try: agent.load(str((repo_root / 'models' / 'dqn' / 'dqn_model_best.pth').resolve()))
        except: print("Could not load DQN model.")
    elif args.agent == 'ddqn':
        agent = DDQNAgent(env.state_dim, env.action_dim)
        # Try multiple common locations for saved models
        loaded = False
        for p in [repo_root / 'models' / 'ddqn' / 'ddqn_model_best.pth', repo_root / 'results' / 'hybrid_results' / 'hybrid_ddqn_model_best.pth', repo_root / 'results' / 'hybrid_results' / 'hybrid_ddqn_model_final.pth']:
            try:
                agent.load(str(p))
                print(f"Loaded DDQN model from {p}")
                loaded = True
                break
            except Exception:
                pass
        if not loaded:
            print("Could not load DDQN model. Proceeding with uninitialized agent.")
    elif args.agent == 'ppo':
        agent = PPOAgent(env.state_dim, env.action_dim)
        loaded = False
        for p in [repo_root / 'models' / 'demo' / 'ppo_demo_model.pth', repo_root / 'results' / 'hybrid_results' / 'hybrid_ppo_model_best.pth', repo_root / 'results' / 'hybrid_results' / 'hybrid_ppo_model_final.pth']:
            try:
                agent.load(str(p))
                print(f"Loaded PPO model from {p}")
                loaded = True
                break
            except Exception:
                pass
        if not loaded:
            print("Could not load PPO model. Proceeding with uninitialized agent.")
    else:
        agent = None

    state = env.reset()
    done = False
    total_reward = 0

    while not done:
        if agent is None:
            action = 0 # default/fixed phase logic
        else:
            action = agent.act(state, eval_mode=True)
            if isinstance(action, tuple):
                action = action[0]
                
        state, reward, done, _ = env.step(action)
        total_reward += reward

    env.close()
    print(f"Visualization complete. Total Reward: {total_reward:.2f}")

if __name__ == "__main__":
    main()
