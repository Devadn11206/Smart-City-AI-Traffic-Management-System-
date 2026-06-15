import os
import subprocess
import sys

def main():
    # Make sure SUMO_HOME is set or we can invoke randomTrips.py from path if it's there
    if 'SUMO_HOME' not in os.environ:
        print("Please set SUMO_HOME environment variable to use randomTrips.py")
        # Trying a fallback if randomTrips.py is in PATH
        script_path = "randomTrips.py" 
    else:
        script_path = os.path.join(os.environ['SUMO_HOME'], 'tools', 'randomTrips.py')
        
    network_file = "../sumo_network/network.net.xml"
    routes_file = "../sumo_network/random_routes.rou.xml"
    
    # Generate background random traffic
    cmd = [
        sys.executable, script_path,
        "-n", network_file,
        "-o", "../sumo_network/random_trips.trips.xml",
        "-e", "3600",
        "-p", "2", # 1 vehicle every 2 seconds -> 1800 vehicles
        "--trip-attributes", 'type="all_vtypes"',
        "--prefix", "rand_veh"
    ]
    
    try:
        print(f"Executing: {' '.join(cmd)}")
        subprocess.run(cmd, check=True)
        print("Random passenger routing generated.")
    except Exception as e:
        print(f"Error running randomTrips.py: {e}")

if __name__ == "__main__":
    main()