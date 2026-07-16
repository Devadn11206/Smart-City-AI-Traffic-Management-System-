from .dqn import DQNAgent

class IDQNAgent:
    def __init__(self, agents_ids, state_dim=10, action_dim=4, lr=1e-3, gamma=0.99, buffer_capacity=50000, batch_size=64):
        self.agents = {
            agent_id: DQNAgent(state_dim, action_dim, lr, gamma, buffer_capacity, batch_size)
            for agent_id in agents_ids
        }
        self.epsilon = 1.0
        
    def act(self, states, evaluate=False):
        # states is a dict: {'TL1': state1, 'TL2': state2, ...}
        return {agent_id: self.agents[agent_id].act(states[agent_id], evaluate) for agent_id in states}
        
    def step(self, states, actions, rewards, next_states, dones):
        for agent_id in states:
            self.agents[agent_id].step(
                states[agent_id], actions[agent_id], rewards[agent_id], next_states[agent_id], dones[agent_id]
            )
            
    def train(self):
        losses = {}
        for agent_id, agent in self.agents.items():
            loss = agent.train()
            losses[agent_id] = loss
        return losses
        
    def update_target(self):
        for agent in self.agents.values():
            agent.update_target()
            
    def decay_epsilon(self):
        for agent in self.agents.values():
            agent.decay_epsilon()
        self.epsilon = list(self.agents.values())[0].epsilon
        
    def save(self, filepath):
        for agent_id, agent in self.agents.items():
            agent_file = filepath.replace(".pth", f"_{agent_id}.pth")
            agent.save(agent_file)
            
    def load(self, filepath):
        for agent_id, agent in self.agents.items():
            agent_file = filepath.replace(".pth", f"_{agent_id}.pth")
            agent.load(agent_file)
        self.epsilon = list(self.agents.values())[0].epsilon
