import os
import json
import time
import argparse
import sys

# Ensure project root is on sys.path so imports like `from utils...` work
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# Import monitoring and env/agent modules from project
from utils.monitoring import start_monitoring
from utils.checkpoint_registry import ensure_canonical_checkpoints, resolve_active_checkpoint
from utils.telemetry import TelemetryCollector
from integration.hybrid_env import HybridSumoEnv
from rl_agents.dqn_agent import DQNAgent
from rl_agents.ddqn_agent import DDQNAgent
from rl_agents.ppo_agent import PPOAgent


def find_model_path(agent_type):
    try:
        return str(resolve_active_checkpoint(agent_type))
    except Exception:
        return None


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--agent', choices=['dqn','ddqn','ppo'], required=True)
    parser.add_argument('--config', default='sumo_network/config.sumocfg')
    parser.add_argument('--pinn', default='pinn_models/trained_pinn.pth')
    parser.add_argument('--duration', type=int, default=600)
    parser.add_argument('--seed', type=int, default=101)
    parser.add_argument('--out-dir', default='results/aco_results/checkpoints')
    parser.add_argument('--gui', action='store_true', default=False, help='Launch SUMO-GUI instead of SUMO')
    args = parser.parse_args()


    outfile_json = os.path.join(args.out_dir, f'hybrid_{args.agent}_seed{args.seed}_metrics_latest.json')
    outfile_log = os.path.join('results', 'logs', f'hybrid_{args.agent}_seed{args.seed}_{int(time.time())}.log')
    os.makedirs(args.out_dir, exist_ok=True)
    os.makedirs(os.path.dirname(outfile_log), exist_ok=True)

    ensure_canonical_checkpoints()
    telemetry = TelemetryCollector(os.path.join('results', 'telemetry'), scenario_id=f'hybrid_{args.agent}', run_id=f'seed{args.seed}')

    # Start monitoring in this process
    try:
        start_monitoring(out_dir=os.path.join('results', 'monitoring'), interval=5.0)
    except Exception:
        pass

    env = None
    agent = None
    try:
        env = HybridSumoEnv(args.config, args.pinn, use_gui=bool(args.gui), num_seconds=args.duration, seed=args.seed)

        state_dim = env.state_dim
        action_dim = env.action_dim
        if args.agent == 'dqn':
            agent = DQNAgent(state_dim, action_dim)
        elif args.agent == 'ddqn':
            agent = DDQNAgent(state_dim, action_dim)
        else:
            agent = PPOAgent(state_dim, action_dim)

        model_path = find_model_path(args.agent)
        load_status = 'missing'
        if model_path:
            try:
                agent.load(model_path)
                load_status = 'loaded'
            except Exception as e:
                load_status = f'load_failed: {str(e)}'
        else:
            load_status = 'not_found'

        telemetry.emit('checkpoint', step=0, checkpoint_path=model_path or '', checkpoint_status=load_status, state_dim=state_dim, action_dim=action_dim)

        state = env.reset(seed=args.seed)
        done = False
        steps = 0
        while not done and env.current_step < args.duration:
            if args.agent == 'ppo':
                import torch
                state_tensor = torch.FloatTensor(state).unsqueeze(0).to(agent.device)
                with torch.no_grad():
                    action_probs, _ = agent.policy_old(state_tensor)
                action_probs = torch.nan_to_num(action_probs, nan=0.0, posinf=0.0, neginf=0.0)
                policy_confidence = float(action_probs.max().item())
                action, _ = agent.act(state, eval_mode=True)
                q_value = None
            else:
                import torch
                state_tensor = torch.FloatTensor(state).unsqueeze(0).to(agent.device)
                with torch.no_grad():
                    q_values = torch.nan_to_num(agent.model(state_tensor), nan=0.0, posinf=0.0, neginf=0.0)
                q_value = float(q_values.max().item())
                policy_confidence = float((q_values.max() - torch.topk(q_values, k=min(2, q_values.shape[-1]), dim=-1).values[..., -1]).item()) if q_values.shape[-1] > 1 else float(q_values.max().item())
                action = agent.act(state, eval_mode=True)
            if args.agent == 'ppo':
                pass
            next_state, reward, done, _ = env.step(action)
            step_metrics = env.get_metrics() if hasattr(env, 'get_metrics') else {}
            telemetry.emit(
                'rl',
                step=steps,
                action=int(action),
                reward=float(reward),
                q_value=q_value,
                policy_confidence=policy_confidence,
                queue=float(step_metrics.get('queue', 0.0) or 0.0),
                wait=float(step_metrics.get('wait', 0.0) or 0.0),
                throughput=float(step_metrics.get('throughput', 0.0) or 0.0),
                avg_speed=float(step_metrics.get('avg_speed', 0.0) or 0.0),
            )
            state = next_state
            steps += 1

        metrics = env.get_episode_metrics()
        metrics['checkpoint_status'] = load_status
        metrics['checkpoint_path'] = model_path or ''
        telemetry.emit('summary', step=steps, **metrics, checkpoint_status=load_status, checkpoint_path=model_path or '')
        with open(outfile_json, 'w') as f:
            json.dump(metrics, f, indent=2)
        print('METRICS:', metrics)
    except Exception as e:
        # write failure record
        err = {'error': str(e)}
        with open(outfile_json, 'w') as f:
            json.dump(err, f, indent=2)
        print('ERROR:', e)
        raise
    finally:
        try:
            if env is not None:
                env.close()
        except Exception:
            pass


if __name__ == '__main__':
    main()
