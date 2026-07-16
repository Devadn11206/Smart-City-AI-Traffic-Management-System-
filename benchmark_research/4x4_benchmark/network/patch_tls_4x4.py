import xml.etree.ElementTree as ET
import os

def patch_tls():
    net_file = os.path.join(os.path.dirname(os.path.abspath(__file__)), "net.net.xml")
    tree = ET.parse(net_file)
    root = tree.getroot()
    
    phases = [
        {'duration': '40', 'state': 'GGGrrrrrrrGGGrrrrrrr'}, # Phase 0
        {'duration': '5',  'state': 'yyyrrrrrrryyyrrrrrrr'}, # Phase 1
        {'duration': '15', 'state': 'rrrGGrrrrrrrrGGrrrrr'}, # Phase 2
        {'duration': '5',  'state': 'rrryyrrrrrrrryyrrrrr'}, # Phase 3
        {'duration': '40', 'state': 'rrrrrGGGrrrrrrrGGGrr'}, # Phase 4
        {'duration': '5',  'state': 'rrrrryyyrrrrrrryyyrr'}, # Phase 5
        {'duration': '15', 'state': 'rrrrrrrrGGrrrrrrrrGG'}, # Phase 6
        {'duration': '5',  'state': 'rrrrrrrryyrrrrrrrryy'}  # Phase 7
    ]

    for tl in root.findall('tlLogic'):
        # Remove existing phases
        for p in tl.findall('phase'):
            tl.remove(p)
            
        # Add new 8 phases
        for i, p_data in enumerate(phases):
            ET.SubElement(tl, 'phase', duration=p_data['duration'], state=p_data['state'])
            
    tree.write(net_file, encoding="UTF-8", xml_declaration=True)
    print("Patched TLS in net.net.xml to 8 phases.")

if __name__ == "__main__":
    patch_tls()
