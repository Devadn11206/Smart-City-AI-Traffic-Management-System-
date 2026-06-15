import os
import random
import xml.etree.ElementTree as ET
from xml.dom import minidom

def prettify(elem):
    rough_string = ET.tostring(elem, 'utf-8')
    reparsed = minidom.parseString(rough_string)
    return reparsed.toprettyxml(indent="    ")

def build_routes(seed=None):
    rng = random.Random(seed)
    routes = ET.Element('routes')

    # Vehicle types are now managed in vehicle_types.add.xml and vehicle_generator.py

    # Define some routes specifically for highway, ring road, commercial, residential
    
    # Highway Route
    ET.SubElement(routes, 'route', {'id': 'highway_eastbound', 'edges': 'hw1_to_hw2 hw2_to_hw3 hw3_to_hw4 hw4_to_hw5'})
    ET.SubElement(routes, 'route', {'id': 'highway_westbound', 'edges': 'hw5_to_hw4 hw4_to_hw3 hw3_to_hw2 hw2_to_hw1'})
    
    # Ring Road Routes (circular)
    ET.SubElement(routes, 'route', {'id': 'ring_clockwise', 'edges': 'rr1_to_rr2 rr2_to_rr3 rr3_to_rr4 rr4_to_rr5 rr5_to_rr6 rr6_to_rr1'})
    ET.SubElement(routes, 'route', {'id': 'ring_counterclock', 'edges': 'rr1_to_rr6 rr6_to_rr5 rr5_to_rr4 rr4_to_rr3 rr3_to_rr2 rr2_to_rr1'})

    # Traffic Flows using equally distributed vehicle types
    # Variable flows for multi-scenario training
    hw_flow = rng.randint(800, 1800)
    rr_flow = rng.randint(300, 800)
    
    # Highway flow
    ET.SubElement(routes, 'flow', {'id': 'flow_hw_eb', 'type': 'all_vtypes', 'route': 'highway_eastbound', 'begin': '0', 'end': '3600', 'vehsPerHour': str(hw_flow)})
    ET.SubElement(routes, 'flow', {'id': 'flow_hw_wb', 'type': 'all_vtypes', 'route': 'highway_westbound', 'begin': '0', 'end': '3600', 'vehsPerHour': str(hw_flow)})
    ET.SubElement(routes, 'flow', {'id': 'flow_hw_eb_trk', 'type': 'all_vtypes', 'route': 'highway_eastbound', 'begin': '0', 'end': '3600', 'vehsPerHour': str(int(hw_flow * 0.15))})
    
    # Ring road flow
    ET.SubElement(routes, 'flow', {'id': 'flow_rr_cw', 'type': 'all_vtypes', 'route': 'ring_clockwise', 'begin': '0', 'end': '3600', 'vehsPerHour': str(rr_flow)})
    ET.SubElement(routes, 'flow', {'id': 'flow_rr_ccw', 'type': 'all_vtypes', 'route': 'ring_counterclock', 'begin': '0', 'end': '3600', 'vehsPerHour': str(rr_flow)})
    
    # Pre-defined Emergency trips (Can just be part of the uniform distribution, but we keep these as guaranteed spawns)
    ET.SubElement(routes, 'trip', {'id': 'amb1', 'type': 'ambulance', 'depart': '100', 'from': 'cz1_to_cz3', 'to': 'hw5_to_rb2'})
    ET.SubElement(routes, 'trip', {'id': 'pol1', 'type': 'police', 'depart': '300', 'from': 'hw1_to_hw2', 'to': 'cz2_to_cz4'})
    ET.SubElement(routes, 'trip', {'id': 'fire1', 'type': 'fire_truck', 'depart': '500', 'from': 'rz1_to_rz3', 'to': 'rr3_to_rr4'})

    # Better path resolving
    current_dir = os.path.dirname(os.path.abspath(__file__))
    output_path = os.path.join(current_dir, '..', 'sumo_network', 'routes.rou.xml')
    output_path = os.path.abspath(output_path)
    
    with open(output_path, 'w') as f:
        f.write(prettify(routes))
        
    print(f"Base Routes generated with Variable Flow (HW: {hw_flow}, RR: {rr_flow}). Seed={seed}")

if __name__ == '__main__':
    build_routes()