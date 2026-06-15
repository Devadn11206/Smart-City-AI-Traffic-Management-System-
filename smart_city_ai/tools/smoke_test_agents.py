import os, sys
ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, ROOT)
from rl_agents.dqn_agent import DQNAgent
from rl_agents.ddqn_agent import DDQNAgent
from rl_agents.ppo_agent import PPOAgent

s = [0.1, 0.0, 0.0, 0.5, 5.0, 0.0, 0.02]

print('Testing DQN...')
d = DQNAgent(7,4)
print('DQN act:', d.act(s, eval_mode=True))
print('Testing DDQN...')
dd = DDQNAgent(7,4)
print('DDQN act:', dd.act(s, eval_mode=True))
print('Testing PPO...')
p = PPOAgent(7,4)
print('PPO act:', p.act(s, eval_mode=True))
print('Agents smoke test done')
