# Traffic Flow Bottleneck Report

Throughput across all three agents remains capped between 330-360 vehicles (vs a baseline potential of 400+).

## 1. Excessive Switching
- **Observation:** PPO achieved up to 84 switches per episode. While this proves the agent is no longer stuck in a 0-switch policy, 84 switches across a 3600-second episode equates to switching the traffic light every 42 seconds on average.
- **Impact:** Each switch introduces a 3-4 second yellow transition phase where no vehicles move (`TimeLoss`). Furthermore, vehicles must decelerate and re-accelerate. The high frequency of switching drastically reduces the total green-time efficiency of the intersection.

## 2. Broken Green Waves
- **Observation:** The environment naturally produces "platoons" of vehicles on the main North-South corridor. A healthy traffic controller will hold the green phase long enough for the entire platoon to pass (a "green wave").
- **Impact:** Because the RL agents are hypersensitive to the Emergency Vehicle wait penalty, they instantly break the green wave the moment an EV spawns on the East-West corridor, trapping the majority of the North-South platoon at a red light.

## 3. Queue Propagation and Congestion
- **Observation:** Because `r_queue_delta` is currently a negligible penalty, the agent does not care if a queue of 30 vehicles builds up on a red phase. 
- **Impact:** These massive queues spill back into upstream intersections (spillback/gridlock). When the agent finally turns the light green for that phase, it takes significantly longer for the queue to discharge because vehicles are densely packed and accelerating from a dead stop.

## 4. Emergency Corridor Paradox
- **Observation:** Emergency delay remains around ~2100 seconds despite the agent heavily prioritizing EVs.
- **Impact:** The agent switches the light green for the EV immediately, but the EV cannot move because the intersection itself is blocked by the congestion the agent created from its erratic switching behavior. The agent successfully minimizes the EV's "red light wait time", but accidentally maximizes the EV's physical traffic jam delay.
