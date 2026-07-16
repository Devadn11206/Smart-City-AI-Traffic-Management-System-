import xml.etree.ElementTree as ET

def patch_tls():
    net_file = "net.net.xml"
    tree = ET.parse(net_file)
    root = tree.getroot()
    
    phases = [
        {'duration': '40', 'state': 'GGGGrrrrrrrrGGGGrrrrrrrr'}, # 0: NS Straight
        {'duration': '5',  'state': 'yyyyrrrrrrrryyyyrrrrrrrr'}, # 1: NS Straight Yellow
        {'duration': '15', 'state': 'rrrrGGrrrrrrrrrrGGrrrrrr'}, # 2: NS Left
        {'duration': '5',  'state': 'rrrryyrrrrrrrrrryyrrrrrr'}, # 3: NS Left Yellow
        {'duration': '40', 'state': 'rrrrrrGGGGrrrrrrrrGGGGrr'}, # 4: EW Straight
        {'duration': '5',  'state': 'rrrrrryyyyrrrrrrrryyyyrr'}, # 5: EW Straight Yellow
        {'duration': '15', 'state': 'rrrrrrrrrrGGrrrrrrrrrrGG'}, # 6: EW Left
        {'duration': '5',  'state': 'rrrrrrrrrryyrrrrrrrrrryy'}  # 7: EW Left Yellow
    ]

    count = 0
    for tl in root.findall('tlLogic'):
        for p in tl.findall('phase'):
            tl.remove(p)
        for p_data in phases:
            ET.SubElement(tl, 'phase', duration=p_data['duration'], state=p_data['state'])
        count += 1
            
    tree.write(net_file, encoding="UTF-8", xml_declaration=True)
    print(f"Patched {count} TLS in net.net.xml to 8 phases.")

if __name__ == "__main__":
    patch_tls()
