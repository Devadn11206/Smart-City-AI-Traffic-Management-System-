import os
import subprocess
import xml.etree.ElementTree as ET
from xml.dom import minidom

def prettify(elem):
    rough_string = ET.tostring(elem, 'utf-8')
    reparsed = minidom.parseString(rough_string)
    return reparsed.toprettyxml(indent="    ")

def build_network():
    # Nodes
    nodes = ET.Element('nodes')
    
    # Adding nodes for a realistic smart city
    
    # 1. Main Highway Nodes
    ET.SubElement(nodes, 'node', {'id': 'hw1', 'x': '-1000', 'y': '0', 'type': 'priority'})
    ET.SubElement(nodes, 'node', {'id': 'hw2', 'x': '-500', 'y': '0', 'type': 'traffic_light'})
    ET.SubElement(nodes, 'node', {'id': 'hw3', 'x': '0', 'y': '0', 'type': 'traffic_light'})
    ET.SubElement(nodes, 'node', {'id': 'hw4', 'x': '500', 'y': '0', 'type': 'traffic_light'})
    ET.SubElement(nodes, 'node', {'id': 'hw5', 'x': '1000', 'y': '0', 'type': 'priority'})
    
    # 2. Ring Road Nodes
    ET.SubElement(nodes, 'node', {'id': 'rr1', 'x': '-800', 'y': '800', 'type': 'priority'})
    ET.SubElement(nodes, 'node', {'id': 'rr2', 'x': '0', 'y': '1000', 'type': 'traffic_light'})
    ET.SubElement(nodes, 'node', {'id': 'rr3', 'x': '800', 'y': '800', 'type': 'priority'})
    ET.SubElement(nodes, 'node', {'id': 'rr4', 'x': '800', 'y': '-800', 'type': 'priority'})
    ET.SubElement(nodes, 'node', {'id': 'rr5', 'x': '0', 'y': '-1000', 'type': 'traffic_light'})
    ET.SubElement(nodes, 'node', {'id': 'rr6', 'x': '-800', 'y': '-800', 'type': 'priority'})
    
    # 3. Commercial Zone Nodes
    ET.SubElement(nodes, 'node', {'id': 'cz1', 'x': '-200', 'y': '300', 'type': 'traffic_light'})
    ET.SubElement(nodes, 'node', {'id': 'cz2', 'x': '200', 'y': '300', 'type': 'traffic_light'})
    ET.SubElement(nodes, 'node', {'id': 'cz3', 'x': '-200', 'y': '600', 'type': 'traffic_light'})
    ET.SubElement(nodes, 'node', {'id': 'cz4', 'x': '200', 'y': '600', 'type': 'traffic_light'})
    
    # 4. Residential Zone Nodes
    ET.SubElement(nodes, 'node', {'id': 'rz1', 'x': '-400', 'y': '-300', 'type': 'priority'})
    ET.SubElement(nodes, 'node', {'id': 'rz2', 'x': '-600', 'y': '-500', 'type': 'priority'})
    ET.SubElement(nodes, 'node', {'id': 'rz3', 'x': '-200', 'y': '-500', 'type': 'priority'})
    
    # 5. Roundabout Nodes
    ET.SubElement(nodes, 'node', {'id': 'rb1', 'x': '400', 'y': '-400', 'type': 'priority'})
    ET.SubElement(nodes, 'node', {'id': 'rb2', 'x': '500', 'y': '-300', 'type': 'priority'})
    ET.SubElement(nodes, 'node', {'id': 'rb3', 'x': '600', 'y': '-400', 'type': 'priority'})
    ET.SubElement(nodes, 'node', {'id': 'rb4', 'x': '500', 'y': '-500', 'type': 'priority'})
    
    # 6. Connecting and T-Junction Nodes
    ET.SubElement(nodes, 'node', {'id': 'tj1', 'x': '-500', 'y': '500', 'type': 'priority'})
    ET.SubElement(nodes, 'node', {'id': 'tj2', 'x': '500', 'y': '500', 'type': 'priority'})
    
    with open('../sumo_network/nodes.nod.xml', 'w') as f:
        f.write(prettify(nodes))
        
    # Edges
    edges = ET.Element('edges')
    
    def add_edge(src, tgt, numLanes, speed, edgetype=""):
        attr = {
            'id': f'{src}_to_{tgt}', 
            'from': src, 
            'to': tgt, 
            'numLanes': str(numLanes), 
            'speed': str(speed)
        }
        if edgetype:
            attr['type'] = edgetype
        ET.SubElement(edges, 'edge', attr)
        
        attr2 = {
            'id': f'{tgt}_to_{src}', 
            'from': tgt, 
            'to': src, 
            'numLanes': str(numLanes), 
            'speed': str(speed)
        }
        if edgetype:
            attr2['type'] = edgetype
        ET.SubElement(edges, 'edge', attr2)

    # Highway edges (high speed, multi-lane)
    hw_speed = 33.33 # ~120 km/h
    add_edge('hw1', 'hw2', 4, hw_speed, 'highway')
    add_edge('hw2', 'hw3', 4, hw_speed, 'highway')
    add_edge('hw3', 'hw4', 4, hw_speed, 'highway')
    add_edge('hw4', 'hw5', 4, hw_speed, 'highway')
    
    # Ring Road (medium-high speed, 2 lanes)
    rr_speed = 22.22 # ~80 km/h
    add_edge('rr1', 'rr2', 2, rr_speed, 'ring')
    add_edge('rr2', 'rr3', 2, rr_speed, 'ring')
    add_edge('rr3', 'rr4', 2, rr_speed, 'ring')
    add_edge('rr4', 'rr5', 2, rr_speed, 'ring')
    add_edge('rr5', 'rr6', 2, rr_speed, 'ring')
    add_edge('rr6', 'rr1', 2, rr_speed, 'ring')
    
    # Connect Ring to Highway
    add_edge('rr1', 'hw1', 2, rr_speed)
    add_edge('rr2', 'hw3', 2, rr_speed)
    add_edge('rr3', 'hw5', 2, rr_speed)
    add_edge('rr6', 'hw1', 2, rr_speed)
    add_edge('rr5', 'hw3', 2, rr_speed)
    add_edge('rr4', 'hw5', 2, rr_speed)
    
    # Commercial Zone (medium speed, 3 lanes)
    cz_speed = 13.89 # ~50 km/h
    add_edge('cz1', 'cz2', 3, cz_speed)
    add_edge('cz3', 'cz4', 3, cz_speed)
    add_edge('cz1', 'cz3', 3, cz_speed)
    add_edge('cz2', 'cz4', 3, cz_speed)
    add_edge('cz1', 'hw2', 3, cz_speed)
    add_edge('cz2', 'hw3', 3, cz_speed)
    add_edge('cz3', 'rr2', 3, cz_speed)
    
    # Residential Zone (low speed, narrow roads 1-2 lanes)
    rz_speed = 8.33 # ~30 km/h
    add_edge('rz1', 'rz2', 1, rz_speed)
    add_edge('rz1', 'rz3', 2, rz_speed)
    add_edge('rz2', 'rz3', 1, rz_speed)
    add_edge('rz1', 'hw2', 2, rz_speed)
    add_edge('rz3', 'rr5', 2, rz_speed)
    
    # Roundabout (1-way circular edges inside + connecting)
    rb_speed = 11.11 # 40 km/h
    # Roundabout internal edges (one-way counter-clockwise)
    ET.SubElement(edges, 'edge', {'id': 'rb1_to_rb2', 'from': 'rb1', 'to': 'rb2', 'numLanes': '2', 'speed': str(rb_speed)})
    ET.SubElement(edges, 'edge', {'id': 'rb2_to_rb3', 'from': 'rb2', 'to': 'rb3', 'numLanes': '2', 'speed': str(rb_speed)})
    ET.SubElement(edges, 'edge', {'id': 'rb3_to_rb4', 'from': 'rb3', 'to': 'rb4', 'numLanes': '2', 'speed': str(rb_speed)})
    ET.SubElement(edges, 'edge', {'id': 'rb4_to_rb1', 'from': 'rb4', 'to': 'rb1', 'numLanes': '2', 'speed': str(rb_speed)})
    
    # Connect Roundabout
    add_edge('hw4', 'rb1', 2, rb_speed)
    add_edge('rr4', 'rb3', 2, rb_speed)
    add_edge('rr5', 'rb4', 2, rb_speed)
    add_edge('hw5', 'rb2', 2, rb_speed)
    
    # T-Junctions
    add_edge('tj1', 'hw2', 2, cz_speed)
    add_edge('tj1', 'cz1', 2, cz_speed)
    add_edge('tj1', 'rr1', 2, cz_speed)
    
    add_edge('tj2', 'hw4', 2, cz_speed)
    add_edge('tj2', 'cz2', 2, cz_speed)
    add_edge('tj2', 'rr3', 2, cz_speed)

    # Some curved edges using shape
    ET.SubElement(edges, 'edge', {
        'id': 'cz4_to_rr3_curve', 
        'from': 'cz4', 
        'to': 'rr3', 
        'numLanes': '2', 
        'speed': str(cz_speed),
        'shape': '200,600 500,700 800,800'
    })
    ET.SubElement(edges, 'edge', {
        'id': 'rr3_to_cz4_curve', 
        'from': 'rr3', 
        'to': 'cz4', 
        'numLanes': '2', 
        'speed': str(cz_speed),
        'shape': '800,800 500,700 200,600'
    })


    with open('../sumo_network/edges.edg.xml', 'w') as f:
        f.write(prettify(edges))
        
    print("Nodes and Edges generated.")

    # Call netconvert to generate the network
    netconvert_cmd = [
        "netconvert",
        "--node-files", "../sumo_network/nodes.nod.xml",
        "--edge-files", "../sumo_network/edges.edg.xml",
        "--output-file", "../sumo_network/network.net.xml",
        "--roundabouts.guess", "true",
        "--ramps.guess", "true",
        "--junctions.join", "true",
        "--tls.guess", "true",
        "--tls.default-type", "static"
    ]
    
    try:
        subprocess.run(netconvert_cmd, check=True)
        print("Network generation complete!")
    except FileNotFoundError:
        print("netconvert not found in path! Please ensure SUMO is installed and added to PATH.")
    except Exception as e:
        print(f"Error running netconvert: {e}")

if __name__ == '__main__':
    build_network()
