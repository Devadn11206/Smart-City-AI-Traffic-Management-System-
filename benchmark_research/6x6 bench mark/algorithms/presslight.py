from .dqn import DQNAgent

class PressLightAgent(DQNAgent):
    """
    PressLight implementation exactly following the original research paper.
    
    Pressure Definition: Incoming Lane Vehicles - Outgoing Lane Vehicles
    State Definition: Number of vehicles on incoming and outgoing lanes + current phase.
                      (In our implementation, we can use the environment's state and pressure infos)
    Action Definition: Traffic light phase (4 discrete actions).
    Reward Definition: -1 * absolute(Pressure)
    
    Why pressure works: It balances the traffic distribution across the network. By minimizing pressure,
    we ensure vehicles don't pile up on incoming lanes while outgoing lanes are empty.
    How it differs: Focuses purely on pressure rather than heuristic waiting times or queue lengths.
    """
    def __init__(self, state_dim=10, action_dim=4, lr=1e-3, gamma=0.99, buffer_capacity=50000, batch_size=64):
        # We reuse the DQN logic. The training script will feed it the PressLight specific reward (-pressure).
        super().__init__(state_dim, action_dim, lr, gamma, buffer_capacity, batch_size)
