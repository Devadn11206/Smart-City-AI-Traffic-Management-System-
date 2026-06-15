import xml.etree.ElementTree as ET
from xml.dom import minidom

def prettify(elem):
    rough_string = ET.tostring(elem, 'utf-8')
    reparsed = minidom.parseString(rough_string)
    return reparsed.toprettyxml(indent="    ")

def generate_vtypes():
    additional = ET.Element('additional')
    
    # Define realistic vehicle types matching the requested list
    vtypes = [
        # Passenger Vehicles
        {'id': 'car', 'vClass': 'passenger', 'color': 'yellow', 'maxSpeed': '33.33', 'guiShape': 'passenger'},
        {'id': 'suv', 'vClass': 'passenger', 'color': 'white', 'maxSpeed': '33.33', 'guiShape': 'passenger/van'},
        {'id': 'hatchback', 'vClass': 'passenger', 'color': 'cyan', 'maxSpeed': '33.33', 'guiShape': 'passenger/hatchback'},
        {'id': 'sedan', 'vClass': 'passenger', 'color': 'orange', 'maxSpeed': '33.33', 'guiShape': 'passenger/sedan'},
        {'id': 'taxi_cab', 'vClass': 'taxi', 'color': '255,204,0', 'maxSpeed': '33.33', 'guiShape': 'passenger/sedan'},
        {'id': 'ev', 'vClass': 'passenger', 'color': 'green', 'maxSpeed': '33.33', 'guiShape': 'passenger/sedan', 'emissionClass': 'Energy/unknown'},
        {'id': 'hybrid', 'vClass': 'passenger', 'color': '150,255,150', 'maxSpeed': '33.33', 'guiShape': 'passenger/sedan'},
        
        # 2-Wheelers
        {'id': 'bike', 'vClass': 'motorcycle', 'color': 'gray', 'maxSpeed': '25.0', 'guiShape': 'motorcycle'},
        
        # Public Transport
        {'id': 'city_bus', 'vClass': 'bus', 'color': 'blue', 'maxSpeed': '22.22', 'guiShape': 'bus'},
        {'id': 'mini_bus', 'vClass': 'bus', 'color': '0,100,255', 'maxSpeed': '25.0', 'guiShape': 'passenger/van'},
        {'id': 'school_bus', 'vClass': 'bus', 'color': '255,204,0', 'maxSpeed': '22.22', 'guiShape': 'bus'},
        
        # Commercial / Delivery / Freight
        {'id': 'truck', 'vClass': 'truck', 'color': 'red', 'maxSpeed': '22.22', 'guiShape': 'truck'},
        {'id': 'cargo_truck', 'vClass': 'truck', 'color': '180,0,0', 'maxSpeed': '20.0', 'guiShape': 'truck'},
        {'id': 'container_truck', 'vClass': 'trailer', 'color': '139,0,0', 'maxSpeed': '20.0', 'guiShape': 'truck/semitrailer'},
        {'id': 'delivery_van', 'vClass': 'delivery', 'color': '180,180,180', 'maxSpeed': '25.0', 'guiShape': 'passenger/van'},
        {'id': 'logistics_vehicle', 'vClass': 'truck', 'color': '128,0,128', 'maxSpeed': '22.0', 'guiShape': 'truck'},
        
        # Emergency Vehicles
        {'id': 'ambulance', 'vClass': 'emergency', 'color': 'white', 'maxSpeed': '40.0', 'guiShape': 'emergency'},
        {'id': 'police', 'vClass': 'emergency', 'color': 'blue', 'maxSpeed': '45.0', 'guiShape': 'police'},
        {'id': 'fire_truck', 'vClass': 'emergency', 'color': 'red', 'maxSpeed': '35.0', 'guiShape': 'firebrigade'},
    ]
    
    for vt in vtypes:
        ET.SubElement(additional, 'vType', vt)
        
    # Create the vTypeDistribution referencing the defined types
    vtype_list = [vt['id'] for vt in vtypes]
    # In SUMO, we can reference existing types inside a vTypeDistribution by simply listing their ids in vTypes attribute
    vtype_dist = ET.SubElement(additional, 'vTypeDistribution', {
        'id': 'all_vtypes',
        'vTypes': " ".join(vtype_list)
    })

    with open('../sumo_network/vehicle_types.add.xml', 'w') as f:
        f.write(prettify(additional))
        
    print("Vehicle types and distribution (all_vtypes) generated in vehicle_types.add.xml.")

if __name__ == '__main__':
    generate_vtypes()