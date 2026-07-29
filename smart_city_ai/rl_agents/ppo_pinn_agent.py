import torch
import torch.nn.functional as F
import numpy as np
import os
from rl_agents.ppo_agent import PPOAgent

class PPOPinnAgent(PPOAgent):
    """
    PINN-Guided PPO Agent via Policy Distillation.
    Extends standard PPO by adding an auxiliary KL-divergence loss during the 
    training phase, guided by a frozen pre-trained PINN Ensemble.
    """
    def __init__(self, state_dim, action_dim, lr_actor=3e-4, lr_critic=3e-4, gamma=0.99, eps_clip=0.2, k_epochs=10,
                 gae_lambda=0.95, entropy_coef=0.01, max_grad_norm=0.5, batch_size=256, device: str | None = None,
                 pinn_ensemble=None, kl_weight=0.1, temperature=1.0):
        super().__init__(state_dim, action_dim, lr_actor, lr_critic, gamma, eps_clip, k_epochs,
                         gae_lambda, entropy_coef, max_grad_norm, batch_size, device)
        
        self.pinn_ensemble = pinn_ensemble
        self.kl_weight = kl_weight
        self.temperature = temperature
        
    def _compute_pinn_target_probs(self, batch_states):
        """
        Computes the target probability distribution over actions using the Physics-Informed model.
        
        1. How the PINN cost function is computed:
           The PINN predicts the next state for each candidate action. The state representation 
           stores queue elements at indices [2, 3] and wait time elements at indices [4, 5].
           We sum these features from the predicted next state to form a physics-based cost.
           
        2. Why the Softmax transformation is used:
           The raw costs are arbitrary negative or positive numbers. We negate the cost (so lower 
           queue/wait time becomes a higher value) and apply Softmax with a temperature parameter. 
           This converts the deterministic physical costs into a valid probability distribution 
           summing to 1, which represents the PINN's "preferred" actions.
           
        3. How the target distribution is generated:
           By evaluating all 4 candidate actions in parallel, computing their costs, and applying Softmax,
           we construct a shape [batch_size, action_dim] target probability tensor.
        """
        if self.pinn_ensemble is None:
            return None
            
        batch_size = batch_states.shape[0]
        action_dim = self.action_dim
        
        # We will collect costs and variances for each action
        action_costs = torch.zeros((batch_size, action_dim), device=self.device)
        action_variances = torch.zeros((batch_size, action_dim), device=self.device)
        
        # Evaluate all candidate actions via PINN
        for a in range(action_dim):
            # Create action tensor [batch_size] filled with action 'a'
            actions_tensor = torch.full((batch_size,), a, dtype=torch.long, device=self.device)
            
            with torch.no_grad():
                mean_next_states, var = self.pinn_ensemble(batch_states, actions_tensor)
                
            # Physics Cost: sum of predicted queue length (indices 2,3) and wait time (indices 4,5)
            # This is a heuristic mapping based on SumoEnvironment6x6 state definition.
            cost = mean_next_states[:, 2:6].sum(dim=1)
            action_costs[:, a] = cost
            action_variances[:, a] = var
            
        # Softmax over negative costs to assign highest probability to lowest cost
        target_logits = -action_costs / self.temperature
        target_probs = F.softmax(target_logits, dim=-1)
        
        # Calculate Confidence Score based on Ensemble Variance (1.0 = full trust, 0.0 = no trust)
        avg_variance = action_variances.mean(dim=-1)
        # alpha is a scaling factor; higher alpha makes the agent more sensitive to variance
        alpha = 10.0 
        confidence_scores = 1.0 / (1.0 + alpha * avg_variance)
        
        return target_probs, confidence_scores

    def replay(self):
        """
        Overrides the standard PPO replay to incorporate the KL-Divergence penalty.
        
        4. How the KL-divergence integrates into the PPO loss:
           We compute the standard PPO clipped surrogate loss (Actor) and Value loss (Critic).
           We then compute the target probabilities using the PINN.
           The KL-divergence between the Actor's current log probabilities and the PINN's target 
           probabilities is added as an auxiliary penalty term multiplied by `self.kl_weight`.
           This regularizes the policy gradient, forcing it toward physics-optimal actions while 
           strictly preserving on-policy data collection.
        """
        if len(self.memory) == 0:
            return
            
        states = torch.FloatTensor(np.array([m[0] for m in self.memory])).to(self.device)
        actions = torch.LongTensor([m[1] for m in self.memory]).to(self.device)
        old_logprobs = torch.FloatTensor([m[2] for m in self.memory]).to(self.device)
        rewards = [m[3] for m in self.memory]
        dones = [m[4] for m in self.memory]
        
        # Compute values for states using current policy
        with torch.no_grad():
            _, state_values_all = self.policy(states)
        state_values_all = state_values_all.squeeze().cpu().numpy()

        # Generalized Advantage Estimation (GAE)
        advantages = np.zeros(len(rewards), dtype=np.float32)
        last_gae_lam = 0
        for t in reversed(range(len(rewards))):
            if t == len(rewards) - 1:
                next_non_terminal = 1.0 - float(dones[t])
                next_values = state_values_all[t] if next_non_terminal > 0 else 0.0
            else:
                next_non_terminal = 1.0 - float(dones[t+1])
                next_values = state_values_all[t+1]
            delta = rewards[t] + self.gamma * next_values * next_non_terminal - state_values_all[t]
            last_gae_lam = delta + self.gamma * self.gae_lambda * next_non_terminal * last_gae_lam
            advantages[t] = last_gae_lam

        advantages = torch.FloatTensor(advantages).to(self.device)
        returns = advantages + torch.FloatTensor(state_values_all).to(self.device)
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

        for _ in range(self.k_epochs):
            indices = np.arange(len(states))
            np.random.shuffle(indices)
            for start in range(0, len(states), self.batch_size):
                end = start + self.batch_size
                batch_idx = indices[start:end]
                
                batch_states = states[batch_idx]
                batch_actions = actions[batch_idx]
                batch_old_logprobs = old_logprobs[batch_idx]
                batch_advantages = advantages[batch_idx]
                batch_returns = returns[batch_idx]
                
                # Actor-Critic Forward Pass
                action_probs, state_values = self.policy(batch_states)
                action_probs = torch.nan_to_num(action_probs, nan=0.0, posinf=0.0, neginf=0.0)
                action_probs = torch.clamp(action_probs, min=1e-8)
                action_probs = action_probs / action_probs.sum(dim=-1, keepdim=True).clamp_min(1e-8)
                
                dist = torch.distributions.Categorical(action_probs)
                logprobs = dist.log_prob(batch_actions)
                dist_entropy = dist.entropy().mean()

                state_values = state_values.squeeze()
                if state_values.dim() == 0:
                    state_values = state_values.unsqueeze(0)

                # Standard PPO Clipping Objective
                ratios = torch.exp(logprobs - batch_old_logprobs.detach())
                surr1 = ratios * batch_advantages
                surr2 = torch.clamp(ratios, 1 - self.eps_clip, 1 + self.eps_clip) * batch_advantages
                actor_loss = -torch.min(surr1, surr2).mean()
                
                # Critic Objective
                critic_loss = 0.5 * self.MseLoss(state_values, batch_returns)
                entropy_loss = -self.entropy_coef * dist_entropy

                # --- NEW: Uncertainty-Aware PINN Policy Distillation (Auxiliary KL-Loss) ---
                kl_loss = 0.0
                if self.pinn_ensemble is not None and self.kl_weight > 0.0:
                    pinn_target_probs, confidence_scores = self._compute_pinn_target_probs(batch_states)
                    log_actor_probs = torch.log(action_probs)
                    
                    # Calculate raw KL divergence per sample [batch_size]
                    raw_kl = F.kl_div(log_actor_probs, pinn_target_probs.detach(), reduction='none').sum(dim=-1)
                    
                    # Dynamically scale the KL penalty using the PINN's uncertainty (confidence_score)
                    kl_loss = (raw_kl * confidence_scores.detach()).mean()
                # ---------------------------------------------------------

                # Total Loss Equation
                loss = actor_loss + critic_loss + entropy_loss + (self.kl_weight * kl_loss)
                self.last_loss = loss.item()

                self.optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.policy.parameters(), max_norm=self.max_grad_norm)
                self.optimizer.step()

        self.policy_old.load_state_dict(self.policy.state_dict())
        self.memory.clear()
