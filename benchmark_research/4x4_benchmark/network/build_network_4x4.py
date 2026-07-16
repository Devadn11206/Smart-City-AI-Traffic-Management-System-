import os
import subprocess

def build_network():
    os.makedirs(os.path.dirname(os.path.abspath(__file__)), exist_ok=True)
    
    grid_size = 4
    spacing = 300
    
    # Generate Nodes
    nodes_xml = ['<nodes>']
    
    # Internal TL nodes
    # TL1 is top-left. Let's make row 1 top, row 4 bottom.
    # row 1: TL1..TL4 (y = 900)
    # row 2: TL5..TL8 (y = 600)
    # row 3: TL9..TL12 (y = 300)
    # row 4: TL13..TL16 (y = 0)
    for row in range(grid_size):
        for col in range(grid_size):
            node_id = f"TL{row * grid_size + col + 1}"
            x = col * spacing
            y = (grid_size - 1 - row) * spacing
            nodes_xml.append(f'    <node id="{node_id}" x="{x}" y="{y}" type="traffic_light"/>')
            
    # Peripheral nodes
    for col in range(grid_size):
        x = col * spacing
        # top
        nodes_xml.append(f'    <node id="top{col+1}" x="{x}" y="{grid_size * spacing}" type="priority"/>')
        # bottom
        nodes_xml.append(f'    <node id="bot{col+1}" x="{x}" y="-{spacing}" type="priority"/>')
        
    for row in range(grid_size):
        y = (grid_size - 1 - row) * spacing
        # left
        nodes_xml.append(f'    <node id="left{row+1}" x="-{spacing}" y="{y}" type="priority"/>')
        # right
        nodes_xml.append(f'    <node id="right{row+1}" x="{grid_size * spacing}" y="{y}" type="priority"/>')
        
    nodes_xml.append('</nodes>')
    
    with open('net.nod.xml', 'w') as f:
        f.write('\n'.join(nodes_xml))
        
    # Generate Edges
    edges_xml = ['<edges>']
    num_lanes = 2
    speed = 13.89
    
    def add_edge(from_node, to_node):
        edges_xml.append(f'    <edge id="{from_node}_{to_node}" from="{from_node}" to="{to_node}" numLanes="{num_lanes}" speed="{speed}"/>')
        edges_xml.append(f'    <edge id="{to_node}_{from_node}" from="{to_node}" to="{from_node}" numLanes="{num_lanes}" speed="{speed}"/>')

    # Internal edges
    for row in range(grid_size):
        for col in range(grid_size):
            curr_id = f"TL{row * grid_size + col + 1}"
            # Connect to right (if not last col)
            if col < grid_size - 1:
                right_id = f"TL{row * grid_size + col + 2}"
                add_edge(curr_id, right_id)
            # Connect to bottom (if not last row)
            if row < grid_size - 1:
                bottom_id = f"TL{(row + 1) * grid_size + col + 1}"
                add_edge(curr_id, bottom_id)
                
    # Peripheral edges
    for col in range(grid_size):
        top_tl = f"TL{col + 1}"
        top_periph = f"top{col + 1}"
        add_edge(top_periph, top_tl)
        
        bot_tl = f"TL{3 * grid_size + col + 1}"
        bot_periph = f"bot{col + 1}"
        add_edge(bot_periph, bot_tl)
        
    for row in range(grid_size):
        left_tl = f"TL{row * grid_size + 1}"
        left_periph = f"left{row + 1}"
        add_edge(left_periph, left_tl)
        
        right_tl = f"TL{row * grid_size + 4}"
        right_periph = f"right{row + 1}"
        add_edge(right_periph, right_tl)

    edges_xml.append('</edges>')
    
    with open('net.edg.xml', 'w') as f:
        f.write('\n'.join(edges_xml))
        
    # Run netconvert
    print("Compiling network with netconvert...")
    subprocess.run([
        'netconvert',
        '--node-files=net.nod.xml',
        '--edge-files=net.edg.xml',
        '--output-file=net.net.xml',
        '--tls.guess', 'true',
        '--tls.default-type', 'static'
    ], check=True)
    print("Network compilation complete: net.net.xml generated.")

if __name__ == "__main__":
    build_network()
