import os

nodes = """<nodes>
    <node id="n1" x="0" y="300" type="traffic_light"/>
    <node id="n2" x="300" y="300" type="traffic_light"/>
    <node id="n3" x="0" y="0" type="traffic_light"/>
    <node id="n4" x="300" y="0" type="traffic_light"/>

    <node id="top1" x="0" y="600" type="priority"/>
    <node id="top2" x="300" y="600" type="priority"/>
    <node id="bot1" x="0" y="-300" type="priority"/>
    <node id="bot2" x="300" y="-300" type="priority"/>
    <node id="left1" x="-300" y="300" type="priority"/>
    <node id="left2" x="-300" y="0" type="priority"/>
    <node id="right1" x="600" y="300" type="priority"/>
    <node id="right2" x="600" y="0" type="priority"/>
</nodes>"""

edges = """<edges>
    <!-- Internal grid edges -->
    <edge id="n1_n2" from="n1" to="n2" numLanes="2" speed="13.89"/>
    <edge id="n2_n1" from="n2" to="n1" numLanes="2" speed="13.89"/>
    <edge id="n1_n3" from="n1" to="n3" numLanes="2" speed="13.89"/>
    <edge id="n3_n1" from="n3" to="n1" numLanes="2" speed="13.89"/>
    <edge id="n2_n4" from="n2" to="n4" numLanes="2" speed="13.89"/>
    <edge id="n4_n2" from="n4" to="n2" numLanes="2" speed="13.89"/>
    <edge id="n3_n4" from="n3" to="n4" numLanes="2" speed="13.89"/>
    <edge id="n4_n3" from="n4" to="n3" numLanes="2" speed="13.89"/>

    <!-- Peripheral edges -->
    <edge id="top1_n1" from="top1" to="n1" numLanes="2" speed="13.89"/>
    <edge id="n1_top1" from="n1" to="top1" numLanes="2" speed="13.89"/>
    <edge id="top2_n2" from="top2" to="n2" numLanes="2" speed="13.89"/>
    <edge id="n2_top2" from="n2" to="top2" numLanes="2" speed="13.89"/>
    
    <edge id="bot1_n3" from="bot1" to="n3" numLanes="2" speed="13.89"/>
    <edge id="n3_bot1" from="n3" to="bot1" numLanes="2" speed="13.89"/>
    <edge id="bot2_n4" from="bot2" to="n4" numLanes="2" speed="13.89"/>
    <edge id="n4_bot2" from="n4" to="bot2" numLanes="2" speed="13.89"/>

    <edge id="left1_n1" from="left1" to="n1" numLanes="2" speed="13.89"/>
    <edge id="n1_left1" from="n1" to="left1" numLanes="2" speed="13.89"/>
    <edge id="left2_n3" from="left2" to="n3" numLanes="2" speed="13.89"/>
    <edge id="n3_left2" from="n3" to="left2" numLanes="2" speed="13.89"/>

    <edge id="right1_n2" from="right1" to="n2" numLanes="2" speed="13.89"/>
    <edge id="n2_right1" from="n2" to="right1" numLanes="2" speed="13.89"/>
    <edge id="right2_n4" from="right2" to="n4" numLanes="2" speed="13.89"/>
    <edge id="n4_right2" from="n4" to="right2" numLanes="2" speed="13.89"/>
</edges>"""

connections = """<connections>
</connections>"""

tll = """<tlLogics>
</tlLogics>"""

def main():
    os.makedirs("benchmark_research/network", exist_ok=True)
    with open("benchmark_research/network/net.nod.xml", "w") as f:
        f.write(nodes)
    with open("benchmark_research/network/net.edg.xml", "w") as f:
        f.write(edges)
    
    # We will use netconvert with --tls.guess to automatically create standard 8-phase traffic lights.
    # Then we can extract the exact phase structure from the generated net.net.xml if needed.

if __name__ == "__main__":
    main()
