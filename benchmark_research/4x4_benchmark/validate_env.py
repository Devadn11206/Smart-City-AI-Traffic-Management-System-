import sys
import os
import numpy as np

# Add paths
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), 'env'))

from sumo_env_4x4 import SumoMultiAgentEnv4x4

def get_colight_adjacency_matrix(grid_size=4):
    """
    Build adjacency matrix for CoLight.
    Each node connects to itself and its direct grid neighbors (up, down, left, right).
    Nodes are numbered 1 to 16, row by row.
    """
    n_nodes = grid_size * grid_size
    adj = np.zeros((n_nodes, n_nodes))
    
    for row in range(grid_size):
        for col in range(grid_size):
            idx = row * grid_size + col
            # Self connection
            adj[idx, idx] = 1
            
            # Up
            if row > 0:
                up_idx = (row - 1) * grid_size + col
                adj[idx, up_idx] = 1
            # Down
            if row < grid_size - 1:
                down_idx = (row + 1) * grid_size + col
                adj[idx, down_idx] = 1
            # Left
            if col > 0:
                left_idx = row * grid_size + (col - 1)
                adj[idx, left_idx] = 1
            # Right
            if col < grid_size - 1:
                right_idx = row * grid_size + (col + 1)
                adj[idx, right_idx] = 1
                
    return adj

def main():
    print("Validating 4x4 Network and Environment...")
    
    base_dir = os.path.dirname(os.path.abspath(__file__))
    net_file = os.path.join(base_dir, 'network', 'net.net.xml')
    route_file = os.path.join(base_dir, 'network', 'routes.rou.xml')
    
    if not os.path.exists(net_file):
        print(f"Error: Network file missing: {net_file}")
        return
    if not os.path.exists(route_file):
        print(f"Error: Route file missing: {route_file}")
        return
        
    env = SumoMultiAgentEnv4x4(net_file, route_file, use_gui=False, num_seconds=300)
    
    print("\nEnvironment initialization successful.")
    print(f"Number of agents (intersections): {len(env.ts_ids)}")
    print(f"Agent IDs: {env.ts_ids}")
    
    print("\nRunning test simulation steps...")
    try:
        states, _ = env.reset()
        print(f"Initial state keys count: {len(states)}")
        
        # Take 5 random steps
        for i in range(5):
            actions = {ts: env.action_space.sample() for ts in env.ts_ids}
            next_states, rewards, dones, _, infos = env.step(actions)
            
        print("Simulation ran 5 steps successfully without errors.")
        print(f"Sample Reward (TL1): {rewards['TL1']:.4f}")
        print(f"Sample Info (TL1): {infos['TL1']}")
    except Exception as e:
        print(f"Simulation Error: {e}")
    finally:
        env.close()
        
    print("\n--- CoLight Adjacency Matrix (16x16) ---")
    adj = get_colight_adjacency_matrix(4)
    print("Rows/Cols represent TL1 through TL16.")
    
    # Print formatted matrix
    for row in adj:
        print("[" + " ".join([str(int(x)) for x in row]) + "]")
        
    print("\nValidation Complete.")

if __name__ == '__main__':
    main()
