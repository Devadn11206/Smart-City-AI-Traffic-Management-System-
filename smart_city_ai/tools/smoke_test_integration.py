import os
import time
import random
import sys
import pathlib
ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, ROOT)
from environment.sumo_environment import SumoEnvironment

CONFIG = os.path.join(ROOT, 'sumo_network', 'config.sumocfg')
print('SMOKE TEST: CONFIG', CONFIG)

def test_env_steps():
    env = SumoEnvironment(CONFIG, use_gui=False, num_seconds=200, action_repeat=1)
    try:
        s = env.reset()
        print('Reset state shape:', s.shape)
        for i in range(10):
            a = random.randint(0, 1)
            ns, r, done, _ = env.step(a)
            print(f'step {i} -> reward {r:.4f} done={done}')
            if done:
                break
        print('Episode metrics:', env.get_episode_metrics())
    finally:
        env.close()


if __name__ == '__main__':
    test_env_steps()
    print('SMOKE TEST: DONE')
