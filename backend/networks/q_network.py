"""
Neural network architectures for all DRL agents.
Q-Networks, Policy Networks, Value Networks, Dueling architecture.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List


class QNetwork(nn.Module):
    """Standard Q-Network: state → Q-values for all actions."""

    def __init__(self, state_dim: int, action_dim: int, hidden_dims: List[int] = None):
        super().__init__()
        hidden_dims = hidden_dims or [256, 256, 128]
        layers = []
        prev = state_dim
        for h in hidden_dims:
            layers.extend([nn.Linear(prev, h), nn.ReLU()])
            prev = h
        layers.append(nn.Linear(prev, action_dim))
        self.net = nn.Sequential(*layers)
        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                nn.init.zeros_(m.bias)

    def forward(self, state: torch.Tensor) -> torch.Tensor:
        return self.net(state)


class DuelingQNetwork(nn.Module):
    """Dueling DQN: separate value and advantage streams."""

    def __init__(self, state_dim: int, action_dim: int, hidden_dims: List[int] = None):
        super().__init__()
        hidden_dims = hidden_dims or [256, 256, 128]
        # Shared feature layers
        shared_layers = []
        prev = state_dim
        for h in hidden_dims[:-1]:
            shared_layers.extend([nn.Linear(prev, h), nn.ReLU()])
            prev = h
        self.shared = nn.Sequential(*shared_layers)

        # Value stream
        self.value_stream = nn.Sequential(
            nn.Linear(prev, hidden_dims[-1]), nn.ReLU(),
            nn.Linear(hidden_dims[-1], 1)
        )
        # Advantage stream
        self.advantage_stream = nn.Sequential(
            nn.Linear(prev, hidden_dims[-1]), nn.ReLU(),
            nn.Linear(hidden_dims[-1], action_dim)
        )
        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                nn.init.zeros_(m.bias)

    def forward(self, state: torch.Tensor) -> torch.Tensor:
        features = self.shared(state)
        value = self.value_stream(features)
        advantage = self.advantage_stream(features)
        q = value + advantage - advantage.mean(dim=-1, keepdim=True)
        return q


class PolicyNetwork(nn.Module):
    """Actor network for policy gradient methods (discrete actions)."""

    def __init__(self, state_dim: int, action_dim: int, hidden_dims: List[int] = None):
        super().__init__()
        hidden_dims = hidden_dims or [256, 256, 128]
        layers = []
        prev = state_dim
        for h in hidden_dims:
            layers.extend([nn.Linear(prev, h), nn.ReLU()])
            prev = h
        layers.append(nn.Linear(prev, action_dim))
        self.net = nn.Sequential(*layers)
        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                nn.init.zeros_(m.bias)

    def forward(self, state: torch.Tensor) -> torch.Tensor:
        logits = self.net(state)
        return F.softmax(logits, dim=-1)

    def get_action_and_log_prob(self, state: torch.Tensor):
        probs = self.forward(state)
        dist = torch.distributions.Categorical(probs)
        action = dist.sample()
        log_prob = dist.log_prob(action)
        entropy = dist.entropy()
        return action, log_prob, entropy


class ValueNetwork(nn.Module):
    """Critic network: state → value estimate."""

    def __init__(self, state_dim: int, hidden_dims: List[int] = None):
        super().__init__()
        hidden_dims = hidden_dims or [256, 256, 128]
        layers = []
        prev = state_dim
        for h in hidden_dims:
            layers.extend([nn.Linear(prev, h), nn.ReLU()])
            prev = h
        layers.append(nn.Linear(prev, 1))
        self.net = nn.Sequential(*layers)
        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                nn.init.zeros_(m.bias)

    def forward(self, state: torch.Tensor) -> torch.Tensor:
        return self.net(state).squeeze(-1)
