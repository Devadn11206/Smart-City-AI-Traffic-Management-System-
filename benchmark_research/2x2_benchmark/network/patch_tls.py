import xml.etree.ElementTree as ET

def patch_tls():
    tree = ET.parse("benchmark_research/network/net.net.xml")
    root = tree.getroot()
    
    # We want 8 phases:
    # 0: NS Straight (0,1,2 and 10,11,12 Green) -> GGGrrrrrrrGGGrrrrrrr
    # 1: NS Straight Yellow -> yyyrrrrrrryyyrrrrrrr
    # 2: NS Left (3,4 and 13,14 Green) -> rrrGGrrrrrrrrGGrrrrr
    # 3: NS Left Yellow -> rrryyrrrrrrrryyrrrrr
    # 4: EW Straight (5,6,7 and 15,16,17 Green) -> rrrrrGGGrrrrrrrGGGrr
    # 5: EW Straight Yellow -> rrrrryyyrrrrrrryyyrr
    # 6: EW Left (8,9 and 18,19 Green) -> rrrrrrrrGGrrrrrrrrGG
    # 7: EW Left Yellow -> rrrrrrrryyrrrrrrrryy
    
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
            
    tree.write("benchmark_research/network/net.net.xml", encoding="UTF-8", xml_declaration=True)

if __name__ == "__main__":
    patch_tls()
