from typing import Callable, Optional, Sequence, Tuple, List
import torch
from torch import nn
import torch.nn.functional as F

from cs285.agents.dqn_agent import DQNAgent


class AWACAgent(DQNAgent):
    def __init__(
        self,
        observation_shape: Sequence[int],
        num_actions: int,
        make_actor: Callable[[Tuple[int, ...], int], nn.Module],
        make_actor_optimizer: Callable[[torch.nn.ParameterList], torch.optim.Optimizer],
        temperature: float,
        **kwargs,
    ):
        super().__init__(observation_shape=observation_shape, num_actions=num_actions, **kwargs)

        self.actor = make_actor(observation_shape, num_actions)
        self.actor_optimizer = make_actor_optimizer(self.actor.parameters())
        self.temperature = temperature

    def compute_critic_loss(
        self,
        observations: torch.Tensor,
        actions: torch.Tensor,
        rewards: torch.Tensor,
        next_observations: torch.Tensor,
        dones: torch.Tensor,
    ):
        with torch.no_grad():
            # Compute the actor's action distribution at the next observations
            next_action_distribution = self.actor(next_observations)
            next_action_probs = next_action_distribution.probs  # Assuming it's a Categorical distribution

            # Compute the expected Q-values under the current policy
            next_qa_values = self.target_critic(next_observations)
            next_q_values = torch.sum(next_action_probs * next_qa_values, dim=1)

            # Compute the TD target using the expected Q-values
            target_values = rewards + self.discount * next_q_values * (~dones)
        
        # TODO(student): Compute Q(s, a) and loss similar to DQN
        qa_values = self.critic(observations)

        action_indices = actions.unsqueeze(-1)
        q_values = qa_values.gather(1, action_indices).squeeze(-1)
        assert q_values.shape == target_values.shape

        loss = F.mse_loss(q_values, target_values)


        return (
            loss,
            {
                "critic_loss": loss.item(),
                "q_values": q_values.mean().item(),
                "target_values": target_values.mean().item(),
            },
            {
                "qa_values": qa_values,
                "q_values": q_values,
            },
        )

    def compute_advantage(
        self,
        observations: torch.Tensor,
        actions: torch.Tensor,
        action_dist: Optional[torch.distributions.Categorical] = None,
    ):
        # TODO(student): compute the advantage of the actions compared to E[Q(s, a)]        
        qa_values_all_actions = self.critic(observations)

        # Select Q-values for the actions taken
        action_indices = actions.unsqueeze(-1)
        qa_values = qa_values_all_actions.gather(1, action_indices).squeeze(-1)

        # Estimate V values (expected Q value under the current policy)
        with torch.no_grad():
            action_probs = torch.softmax(self.actor(observations).logits, dim=-1)
            # Compute expected Q values under the current policy
            q_values = torch.sum(qa_values_all_actions * action_probs, dim=1)

        # Compute advantages
        advantages = qa_values - q_values

        return advantages

    def update_actor(
        self,
        observations: torch.Tensor,
        actions: torch.Tensor,
    ):
        # TODO(student): update the actor using AWAC
        categorical_dist = self.actor(observations)
        logits = categorical_dist.logits
        log_probs = torch.log_softmax(logits, dim=-1)
        selected_log_probs = log_probs.gather(1, actions.unsqueeze(-1)).squeeze(-1)
        advantages = self.compute_advantage(observations, actions)
        weights = torch.exp(advantages / self.temperature)
            
        
        loss = -(weights * selected_log_probs).mean()

        self.actor_optimizer.zero_grad()
        loss.backward()
        self.actor_optimizer.step()

        return loss.item()

    def update(self, observations: torch.Tensor, actions: torch.Tensor, rewards: torch.Tensor, next_observations: torch.Tensor, dones: torch.Tensor, step: int):
        metrics = super().update(observations, actions, rewards, next_observations, dones, step)

        # Update the actor.
        actor_loss = self.update_actor(observations, actions)
        metrics["actor_loss"] = actor_loss

        return metrics
