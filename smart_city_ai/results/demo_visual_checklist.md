# Smart City Demo Visual Checklist

Use this checklist during your live demonstration to ensure all systems are visually verifiable and functioning flawlessly in the SUMO GUI.

## Preparation: Best Camera Settings
To make the demo visually impressive and professional, configure your SUMO GUI immediately after it launches but before you hit "Play":
- **Zoom Level:** Zoom in close enough to clearly see individual vehicle movement, but keep the core central intersection fully in view.
- **Simulation Speed:** Set the TraCI delay to **50-100ms** so that the RL traffic light switching and Emergency Vehicle corridor clearance happens at a human-readable pace. Do not run at 0ms delay.
- **Vehicle Coloring Options:** Ensure your SUMO config colors Emergency Vehicles explicitly. Highlight them in **bright RED**.
- **Edge Highlighting:** Watch for the targeted green-wave edges turning green/yellow in the GUI (or watch the traffic signals themselves snap to green well before the EV arrives).
- **Traffic Light States:** Point out the phase changes on the traffic light indicators in the center of the intersection.

## Live Checklist

- `[ ]` **SUMO GUI opens:** Simulation initializes correctly.
- `[ ]` **Vehicles visible:** Civilian traffic (`flow_hw_eb`, `flow_rr_cw`) spawns and moves naturally.
- `[ ]` **Traffic lights visible:** The central intersection shows active red/yellow/green states.
- `[ ]` **PPO model loaded:** The terminal verifies `ppo_demo_model.pth` was successfully loaded by the Hybrid Controller.
- `[ ]` **PINN predictions running:** Terminal logs show density and velocity predictions printing out periodically.
- `[ ]` **Traffic lights change:** The PPO agent begins switching phases dynamically to prevent queue buildup.
- `[ ]` **Emergency vehicle spawns:** At `t=100`, the ambulance (`amb1`) visibly spawns into the network.
- `[ ]` **Emergency detected:** The TraCI telemetry registers the vehicle class as "emergency".
- `[ ]` **Corridor activated:** The RL agent's control is temporarily overridden. The emergency corridor asserts a preemptive Green Wave.
- `[ ]` **Traffic clearing actions:** Civilian vehicles in the EV's path execute lane changes or slow down to part the traffic.
- `[ ]` **Emergency clears intersection:** The EV successfully breezes through the core intersection without stopping.
- `[ ]` **ACO routing visible:** Civilian traffic reroutes to avoid the reserved emergency corridor, dynamically responding to pheromone updates.
- `[ ]` **Metrics generated:** At the end of the simulation, the `results/demo_metrics` folder populates with throughput and wait times.
