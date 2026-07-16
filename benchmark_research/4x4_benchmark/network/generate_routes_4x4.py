import os
import argparse
import random

def generate_routes(congestion_level='Medium'):
    # Probabilities per flow
    if congestion_level == 'Low':
        prob = 0.03
    elif congestion_level == 'High':
        prob = 0.15
    else:
        prob = 0.08

    # Entering edges
    top_in = [f"top{i}_TL{i}" for i in range(1, 5)]
    bot_in = [f"bot{i}_TL{12+i}" for i in range(1, 5)]
    left_in = [f"left{i}_TL{1+(i-1)*4}" for i in range(1, 5)]
    right_in = [f"right{i}_TL{i*4}" for i in range(1, 5)]

    # Exiting edges
    top_out = [f"TL{i}_top{i}" for i in range(1, 5)]
    bot_out = [f"TL{12+i}_bot{i}" for i in range(1, 5)]
    left_out = [f"TL{1+(i-1)*4}_left{i}" for i in range(1, 5)]
    right_out = [f"TL{i*4}_right{i}" for i in range(1, 5)]
    
    xml = ['<routes>', '    <vType id="car" accel="2.6" decel="4.5" sigma="0.5" length="5" minGap="2.5" maxSpeed="15" color="1,1,0"/>']

    flow_id = 0
    
    def add_flows(sources, destinations, label):
        nonlocal flow_id
        for src in sources:
            # Pick a few random destinations for each source to avoid explosion of flows
            # but ensure cross traffic.
            targets = random.sample(destinations, k=min(2, len(destinations)))
            for dst in targets:
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
    parser.add_argument("--level", type=str, default="Medium", choices=["Low", "Medium", "High"])
    args = parser.parse_args()
    generate_routes(args.level)
