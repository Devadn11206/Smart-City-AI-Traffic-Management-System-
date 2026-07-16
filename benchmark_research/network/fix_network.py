import re
import os

def fix_network():
    net_path = os.path.join(os.path.dirname(__file__), 'net.net.xml')
    with open(net_path, 'r') as f:
        content = f.read()

    # We need to replace the 4-phase tlLogic with an 8-phase tlLogic
    # For each TL1, TL2, TL3, TL4
    
    for tl_id in ['TL1', 'TL2', 'TL3', 'TL4']:
        # Regex to find the tlLogic block
        pattern = f'<tlLogic id="{tl_id}" type="static" programID="0" offset="0">.*?</tlLogic>'
        
        replacement = f"""<tlLogic id="{tl_id}" type="static" programID="0" offset="0">
        <phase duration="42" state="GGGGrrrrrrrrGGGGrrrrrrrr"/>
        <phase duration="3"  state="yyyyyyrrrrrryyyyyyrrrrrr"/>
        <phase duration="42" state="rrrrGGrrrrrrrrrrGGrrrrrr"/>
        <phase duration="3"  state="rrrryyrrrrrrrrrryyrrrrrr"/>
        <phase duration="42" state="rrrrrrGGGGrrrrrrrrGGGGrr"/>
        <phase duration="3"  state="rrrrrryyyyyyrrrrrryyyyyy"/>
        <phase duration="42" state="rrrrrrrrrrGGrrrrrrrrrrGG"/>
        <phase duration="3"  state="rrrrrrrrrryyrrrrrrrrrryy"/>
    </tlLogic>"""
        
        content = re.sub(pattern, replacement, content, flags=re.DOTALL)

    with open(net_path, 'w') as f:
        f.write(content)
        
    print("Network fixed successfully!")

if __name__ == '__main__':
    fix_network()
