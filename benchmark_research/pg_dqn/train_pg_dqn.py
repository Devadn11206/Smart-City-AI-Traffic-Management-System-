import os
import sys
import yaml
import time
import numpy as np
import torch

# Paths setup
current_dir = os.path.dirname(os.path.abspath(__file__))
benchmark_dir = os.path.dirname(current_dir)
if benchmark_dir not in sys.path:
    sys.path.append(benchmark_dir)
    
env_dir = os.path.join(benchmark_dir, '6x6_benchmark')
if env_dir not in sys.path:
    sys.path.append(env_dir)

from pg_dqn.dqn_agent import AC_PG_DQNAgent
from pg_dqn.replay_buffer import ReplayBuffer
from pg_dqn.utils.logger import PGDQNLogger
from pg_dqn.utils.checkpoint import CheckpointManager
from env.sumo_env_6x6 import SumoMultiAgentEnv6x6  # type: ignore

def main():
    with open(os.path.join(current_dir, 'config.yaml'), 'r') as f:
        config = yaml.safe_load(f)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Init Env
    net_file = os.path.join(benchmark_dir, '6x6_benchmark', 'network', 'net.net.xml')
    route_file = os.path.join(benchmark_dir, '6x6_benchmark', 'network', 'routes.rou.xml')
    env = SumoMultiAgentEnv6x6(net_file=net_file, route_file=route_file, use_gui=False, num_seconds=config['training']['max_steps']*10, reward_weights=config.get('reward_weights'))
    
    # Init Agent
    agent = AC_PG_DQNAgent(
        state_dim=config['environment']['state_dim'],
        action_dim=config['environment']['action_dim'],
        physics_dim=config['environment']['physics_dim'],
        lr=config['training']['lr'],
        gamma=config['training']['gamma'],
        epsilon_start=config['exploration']['epsilon_start'],
        epsilon_end=config['exploration']['epsilon_end'],
        epsilon_decay=config['exploration']['epsilon_decay'],
        device=device,
        ablation_mode=config['training'].get('ablation_mode', 2),
        use_ddqn=config['training'].get('use_ddqn', False)
    )
    
    start_episode = 1
    # Resume from checkpoint if provided
    if len(sys.argv) > 1 and sys.argv[1].endswith('.pth'):
        print(f"Resuming from checkpoint: {sys.argv[1]}")
        checkpoint = torch.load(sys.argv[1], map_location=device)
        if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
            agent.q_network.load_state_dict(checkpoint['model_state_dict'])
            start_episode = checkpoint.get('episode', 0) + 1
            agent.epsilon = checkpoint.get('epsilon', agent.epsilon)
            print(f"Resumed from episode {start_episode-1} with epsilon {agent.epsilon:.2f}")
        else:
            agent.q_network.load_state_dict(checkpoint)
            print("Loaded old-format checkpoint (model weights only).")
        agent.target_network.load_state_dict(agent.q_network.state_dict())        
    replay_buffer = ReplayBuffer(config['training']['buffer_size'])
    
    run_name = f"pg_dqn_run_{int(time.time())}"
    log_dir = os.path.join(benchmark_dir, 'results', 'pg_dqn', run_name)
    logger = PGDQNLogger(log_dir)
    checkpointer = CheckpointManager(log_dir)
    
    global_step = 0
    batch_size = config['training']['batch_size']
    update_freq = config['training']['target_update_freq']
    
    from tqdm import tqdm
    
    for episode in range(start_episode, config['training']['episodes'] + 1):
        states, _ = env.reset()
        episode_reward = 0
        episode_losses = []
        ep_avg_qs = []
        ep_target_qs = []
        ep_td_errors = []
        ep_grad_norms = []
        last_infos = None
        
        ep_start_time = time.time()
        pinn_time = 0.0
        dqn_time = 0.0
        
        done = False
        pbar = tqdm(total=config['training']['max_steps'], desc=f"Episode {episode}")
        
        while not done:
            ts_ids = list(states.keys())
            states_np = np.array([states[ts] for ts in ts_ids])
            
            # 1. Batched PINN Inference
            t0 = time.time()
            pinn_preds = agent.get_pinn_predictions(env, ts_ids, last_infos, episode=episode)
            pinn_time += (time.time() - t0)
            
            # 2. Action Selection
            states_tensor = torch.FloatTensor(states_np).to(device)
            pinn_tensor = torch.FloatTensor(pinn_preds).to(device)
            actions_list, d_time = agent.act(states_tensor, pinn_tensor)
            dqn_time += d_time
            
            actions = {ts_ids[i]: actions_list[i] for i in range(len(ts_ids))}
            
            # 3. Step Environment
            next_states, rewards, dones, _, infos = env.step(actions)
            
            # 4. Batched Next PINN Inference (for the replay buffer)
            t0 = time.time()
            next_pinn_preds = agent.get_pinn_predictions(env, ts_ids, infos, episode=None)
            pinn_time += (time.time() - t0)
            
            # 5. Store Transitions
            for i, ts in enumerate(ts_ids):
                replay_buffer.push(
                    states[ts], 
                    pinn_preds[i], 
                    actions[ts], 
                    rewards[ts], 
                    next_states[ts], 
                    next_pinn_preds[i], 
                    float(dones[ts])
                )
                episode_reward += rewards[ts]
                
            # 6. Train Step
            loss, avg_q, avg_target_q, td_error, grad_norm = agent.train_step(replay_buffer, batch_size)
            if loss > 0:
                episode_losses.append(loss)
                ep_avg_qs.append(avg_q)
                ep_target_qs.append(avg_target_q)
                ep_td_errors.append(td_error)
                ep_grad_norms.append(grad_norm)
                
            # 7. Update Target Network
            global_step += 1
            if global_step % update_freq == 0:
                agent.update_target_network()
                
            states = next_states
            last_infos = infos
            
            if all(dones.values()):
                done = True
                
            pbar.update(1)
                
        pbar.close()
        agent.decay_epsilon()
        train_time = time.time() - ep_start_time
        
        # Metrics
        avg_loss = np.mean(episode_losses) if episode_losses else 0.0
        ep_avg_q_val = np.mean(ep_avg_qs) if ep_avg_qs else 0.0
        ep_target_q_val = np.mean(ep_target_qs) if ep_target_qs else 0.0
        ep_td_err_val = np.mean(ep_td_errors) if ep_td_errors else 0.0
        ep_grad_norm_val = np.mean(ep_grad_norms) if ep_grad_norms else 0.0
        buf_size = len(replay_buffer)
        
        global_info = infos.get('global', {})
        avg_wait = np.mean([infos[ts]['wait'] for ts in ts_ids])
        queue = np.sum([infos[ts]['queue'] for ts in ts_ids])
        tp = np.sum([infos[ts]['throughput'] for ts in ts_ids])
        fuel = global_info.get('fuel_consumption', 0)
        speed = np.mean([infos[ts]['speed'] for ts in ts_ids])
        travel_time = global_info.get('avg_travel_time', 0)
        
        logger.log_episode(
            episode, episode_reward, avg_wait, queue, tp, fuel, speed, travel_time, 
            agent.epsilon, avg_loss, ep_avg_q_val, ep_target_q_val, ep_td_err_val, buf_size, ep_grad_norm_val, train_time, pinn_time, dqn_time
        )
        
        checkpointer.save(agent, episode, episode_reward, avg_wait)
        
        print(f"Ep: {episode} | R: {episode_reward:.2f} | Wait: {avg_wait:.2f} | Eps: {agent.epsilon:.2f} | Time: {train_time:.1f}s")

if __name__ == "__main__":
    main()
