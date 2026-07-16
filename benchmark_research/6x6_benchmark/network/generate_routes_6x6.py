import os
import argparse
import random

def generate_routes(congestion_level='High'):
    # Increased probability for 6x6 scale
    # 3 lanes can handle more capacity, but 6x6 grid can gridlock if too high.
    if congestion_level == 'Low':
        prob = 0.05
    elif congestion_level == 'High':
        prob = 0.15
    else:
        prob = 0.10

    grid_size = 6

    # Entering edges
    top_in = [f"top{i}_TL{i}" for i in range(1, grid_size+1)]
    bot_in = [f"bot{i}_TL{(grid_size-1)*grid_size + i}" for i in range(1, grid_size+1)]
    left_in = [f"left{i}_TL{1+(i-1)*grid_size}" for i in range(1, grid_size+1)]
    right_in = [f"right{i}_TL{i*grid_size}" for i in range(1, grid_size+1)]

    # Exiting edges
    top_out = [f"TL{i}_top{i}" for i in range(1, grid_size+1)]
    bot_out = [f"TL{(grid_size-1)*grid_size + i}_bot{i}" for i in range(1, grid_size+1)]
    left_out = [f"TL{1+(i-1)*grid_size}_left{i}" for i in range(1, grid_size+1)]
    right_out = [f"TL{i*grid_size}_right{i}" for i in range(1, grid_size+1)]
    
    xml = ['<routes>', '    <vType id="car" accel="2.6" decel="4.5" sigma="0.5" length="5" minGap="2.5" maxSpeed="15" color="1,1,0"/>']

    flow_id = 0
    
    # Use fixed seed for deterministic routes in benchmark
    random.seed(42)
    
    def add_flows(sources, destinations, label):
        nonlocal flow_id
        for src in sources:
            # Pick 3 random destinations to spread traffic in 6x6
            targets = random.sample(destinations, k=min(3, len(destinations)))
            for dst in targets:
                # Use randomized routing (jtrrouter isn't used, SUMO will use shortest path)
                xml.append(f'    <flow id="flow_{label}_{flow_id}" type="car" begin="0" end="3600" probability="{prob}" from="{src}" to="{dst}"/>')
                flow_id += 1

    # North to South
    add_flows(top_in, bot_out, "N2S")
    # North to East/West
    add_flows(top_in, left_out + right_out, "N2EW")
    
    # South to North
    add_flows(bot_in, top_out, "S2N")
    # South to East/West
    add_flows(bot_in, left_out + right_out, "S2EW")

    # East to West
    add_flows(right_in, left_out, "E2W")
    # East to North/South
    add_flows(right_in, top_out + bot_out, "E2NS")

    # West to East
    add_flows(left_in, right_out, "W2E")
    # West to North/South
    add_flows(left_in, top_out + bot_out, "W2NS")

    xml.append('</routes>')
    
    output_file = 'routes.rou.xml'
    with open(output_file, 'w') as f:
        f.write('\n'.join(xml))
        
    print(f"Generated {output_file} with {congestion_level} congestion (probability={prob}).")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--level", type=str, default="High", choices=["Low", "Medium", "High"])
    args = parser.parse_args()
    generate_routes(args.level)
