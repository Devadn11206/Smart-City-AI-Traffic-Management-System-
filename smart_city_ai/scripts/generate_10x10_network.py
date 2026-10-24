import os
import subprocess
import sys

def generate_10x10_network():
    base_dir = os.path.dirname(os.path.abspath(__file__))
    # Output to dqn_pinn_package as that's where train_pi_ddqn_6x6 looks for configs usually
    net_dir = os.path.abspath(os.path.join(base_dir, "..", "..", "dqn_pinn_package", "sumo_network_10x10"))
    os.makedirs(net_dir, exist_ok=True)
    
    net_file = os.path.join(net_dir, "10x10.net.xml")
    
    # Generate the 10x10 grid network
    print(f"Generating 10x10 network at {net_file}...")
    subprocess.run([
        "netgenerate", 
        "--grid", 
        "--grid.x-number", "10", 
        "--grid.y-number", "10", 
        "--grid.x-length", "150", 
        "--grid.y-length", "150",
        "--default.lanenumber", "2",
        "--tls.guess", "true",  # Ensure intersections are traffic lights
        "-o", net_file
    ], check=True)
    
    print("Network generated successfully.")
    
    # Generate random trips
    trips_file = os.path.join(net_dir, "random_trips.trips.xml")
    routes_file = os.path.join(net_dir, "routes.rou.xml")
    
    print(f"Generating random routes at {routes_file}...")
    sumo_home = os.environ.get("SUMO_HOME")
    if not sumo_home:
        raise ValueError("SUMO_HOME environment variable is not set.")
        
    random_trips_script = os.path.join(sumo_home, "tools", "randomTrips.py")
    
    subprocess.run([
        "python", random_trips_script,
        "-n", net_file,
        "-o", trips_file,
        "-r", routes_file,
        "-e", "3600",
        "-p", "1.0" # insert one vehicle per second globally
    ], check=True)
    
    print("Routes generated successfully.")
    
    # Create config file
    config_file = os.path.join(net_dir, "config.sumocfg")
    config_content = f"""<?xml version="1.0" encoding="UTF-8"?>
<configuration>
    <input>
        <net-file value="10x10.net.xml"/>
        <route-files value="routes.rou.xml"/>
    </input>
    <time>
        <begin value="0"/>
        <end value="3600"/>
    </time>
</configuration>
"""
    with open(config_file, 'w') as f:
        f.write(config_content)
        
    print(f"Config file generated at {config_file}")

if __name__ == "__main__":
    generate_10x10_network()
