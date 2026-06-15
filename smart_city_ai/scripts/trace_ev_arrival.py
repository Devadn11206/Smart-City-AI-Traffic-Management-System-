import os
import sys
import traci
from pathlib import Path

repo_root = Path(__file__).resolve().parent.parent
if str(repo_root) not in sys.path:
    sys.path.insert(0, str(repo_root))

from environment.sumo_environment import SumoEnvironment
from scripts.generate_routes import build_routes

def trace_emergency_vehicles():
    config_file = repo_root / "sumo_network" / "config.sumocfg"
    build_routes(seed=9999)
    env = SumoEnvironment(str(config_file), use_gui=False, num_seconds=3600)
    
    print("\n--- STARTING DIAGNOSTIC SIMULATION ---")
    state = env.reset(seed=9999)
    done = False
    
    ev_data = {
        'amb1': {'spawn_time': None, 'route_length': None, 'edges': set(), 'final_edge': None, 'arrived': False},
        'pol1': {'spawn_time': None, 'route_length': None, 'edges': set(), 'final_edge': None, 'arrived': False},
        'fire1': {'spawn_time': None, 'route_length': None, 'edges': set(), 'final_edge': None, 'arrived': False}
    }
    
    while not done:
        # Just use simple fixed action 1 to advance
        state, reward, done, _ = env.step(1)
        step = env.current_step
        
        arrived_this_step = traci.simulation.getArrivedIDList()
        current_vehicles = traci.vehicle.getIDList()
        
        for ev in ev_data.keys():
            if ev in current_vehicles:
                if ev_data[ev]['spawn_time'] is None:
                    ev_data[ev]['spawn_time'] = step
                    route = traci.vehicle.getRoute(ev)
                    ev_data[ev]['route_length'] = len(route)
                    
                edge = traci.vehicle.getRoadID(ev)
                if not edge.startswith(":"):
                    ev_data[ev]['edges'].add(edge)
                    ev_data[ev]['final_edge'] = edge
            
            if ev in arrived_this_step:
                print(f"[{step}] SUCCESS: {ev} has ARRIVED!")
                ev_data[ev]['arrived'] = True
    
    env.close()
    
    print("\n--- EMERGENCY VEHICLE TRACE RESULTS ---")
    for ev, data in ev_data.items():
        print(f"Vehicle: {ev}")
        print(f"  Spawned: {data['spawn_time']}")
        print(f"  Route Length: {data['route_length']} edges")
        print(f"  Final Edge: {data['final_edge']}")
        print(f"  Arrived: {data['arrived']}")
        print(f"  Edges Traversed: {len(data['edges'])}")
        print("---------------------------------------")

if __name__ == "__main__":
    trace_emergency_vehicles()
