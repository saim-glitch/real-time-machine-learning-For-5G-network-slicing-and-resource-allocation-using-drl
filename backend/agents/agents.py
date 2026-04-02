"""
DRL Agents for Network Slicing Resource Allocation.
Implements DQN, DDQN, Dueling DQN, PPO, SAC, and baseline algorithms.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
from abc import ABC, abstractmethod
from typing import Dict, Tuple, Optional, List
import copy
import sys, os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from networks.q_network import QNetwork, DuelingQNetwork, PolicyNetwork, ValueNetwork
from replay.replay_buffer import ReplayBuffer, PrioritizedReplayBuffer
from config import DRLConfig


# ═══════════════════════════════════════════════════════════════════════════
# Action Encoding / Decoding
# ═══════════════════════════════════════════════════════════════════════════

def encode_action(multi_action: np.ndarray, num_slices: int = 3) -> int:
    """Convert [0-2, 0-2, 0-2] → single int 0-26."""
    result = 0
    for i, a in enumerate(multi_action):
        result += int(a) * (3 ** i)
    return result

def decode_action(action_idx: int, num_slices: int = 3) -> np.ndarray:
    """Convert single int 0-26 → [0-2, 0-2, 0-2]."""
    result = []
    for _ in range(num_slices):
        result.append(action_idx % 3)
        action_idx //= 3
    return np.array(result, dtype=np.int32)


# ═══════════════════════════════════════════════════════════════════════════
# Base Agent
# ═══════════════════════════════════════════════════════════════════════════

class BaseAgent(ABC):
    """Abstract base class for all DRL agents."""

    def __init__(self, name: str, cfg: DRLConfig = None):
        self.name = name
        self.cfg = cfg or DRLConfig()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.training_step = 0

    @abstractmethod
    def select_action(self, state: np.ndarray, evaluate: bool = False) -> int:
        pass

    @abstractmethod
    def update(self, batch: tuple) -> Dict[str, float]:
        pass

    @abstractmethod
    def save(self, path: str):
        pass

    @abstractmethod
    def load(self, path: str):
        pass


# ═══════════════════════════════════════════════════════════════════════════
# DQN Agent
# ═══════════════════════════════════════════════════════════════════════════

class DQNAgent(BaseAgent):
    """Deep Q-Network with target network and experience replay."""

    def __init__(self, cfg: DRLConfig = None, use_per: bool = False):
        super().__init__("DQN", cfg)
        self.q_net = QNetwork(self.cfg.state_dim, self.cfg.action_dim, self.cfg.hidden_dims).to(self.device)
        self.target_net = copy.deepcopy(self.q_net).to(self.device)
        self.target_net.eval()
        self.optimizer = optim.Adam(self.q_net.parameters(), lr=self.cfg.learning_rate)

        self.use_per = use_per
        if use_per:
            self.buffer = PrioritizedReplayBuffer(
                self.cfg.buffer_size, self.cfg.state_dim,
                alpha=self.cfg.per_alpha, beta_start=self.cfg.per_beta_start
            )
        else:
            self.buffer = ReplayBuffer(self.cfg.buffer_size, self.cfg.state_dim)

        self.epsilon = self.cfg.epsilon_start

    def _update_epsilon(self):
        self.epsilon = max(
            self.cfg.epsilon_end,
            self.cfg.epsilon_start - (self.cfg.epsilon_start - self.cfg.epsilon_end)
            * self.training_step / self.cfg.epsilon_decay_steps
        )

    def select_action(self, state: np.ndarray, evaluate: bool = False) -> int:
        if not evaluate and np.random.random() < self.epsilon:
            return np.random.randint(self.cfg.action_dim)
        with torch.no_grad():
            state_t = torch.FloatTensor(state).unsqueeze(0).to(self.device)
            q_values = self.q_net(state_t)
            return int(q_values.argmax(dim=1).item())

    def store_transition(self, state, action, reward, next_state, done):
        self.buffer.push(state, action, reward, next_state, done)

    def update(self, batch=None) -> Dict[str, float]:
        if len(self.buffer) < self.cfg.batch_size:
            return {"loss": 0.0}

        if batch is None:
            states, actions, rewards, next_states, dones, indices, weights = \
                self.buffer.sample(self.cfg.batch_size)
        else:
            states, actions, rewards, next_states, dones, indices, weights = batch

        states_t = torch.FloatTensor(states).to(self.device)
        actions_t = torch.LongTensor(actions).to(self.device)
        rewards_t = torch.FloatTensor(rewards).to(self.device)
        next_states_t = torch.FloatTensor(next_states).to(self.device)
        dones_t = torch.FloatTensor(dones).to(self.device)
        weights_t = torch.FloatTensor(weights).to(self.device)

        # Current Q-values
        q_values = self.q_net(states_t).gather(1, actions_t.unsqueeze(1)).squeeze(1)

        # Target Q-values
        with torch.no_grad():
            next_q = self.target_net(next_states_t).max(dim=1)[0]
            target = rewards_t + self.cfg.gamma * next_q * (1 - dones_t)

        # Weighted loss
        td_errors = (q_values - target).detach().cpu().numpy()
        loss = (weights_t * F.mse_loss(q_values, target, reduction='none')).mean()

        self.optimizer.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(self.q_net.parameters(), 10.0)
        self.optimizer.step()

        # Update priorities
        if self.use_per:
            self.buffer.update_priorities(indices, td_errors)

        # Update target network
        self.training_step += 1
        self._update_epsilon()
        if self.training_step % self.cfg.target_update_freq == 0:
            self.target_net.load_state_dict(self.q_net.state_dict())

        return {"loss": loss.item(), "q_mean": q_values.mean().item(), "epsilon": self.epsilon}

    def save(self, path: str):
        torch.save({"q_net": self.q_net.state_dict(), "step": self.training_step}, path)

    def load(self, path: str):
        ckpt = torch.load(path, map_location=self.device, weights_only=True)
        self.q_net.load_state_dict(ckpt["q_net"])
        self.target_net.load_state_dict(ckpt["q_net"])
        self.training_step = ckpt.get("step", 0)


# ═══════════════════════════════════════════════════════════════════════════
# Double DQN Agent
# ═══════════════════════════════════════════════════════════════════════════

class DDQNAgent(DQNAgent):
    """Double DQN — reduces Q-value overestimation."""

    def __init__(self, cfg: DRLConfig = None, use_per: bool = True):
        super().__init__(cfg, use_per)
        self.name = "DDQN"

    def update(self, batch=None) -> Dict[str, float]:
        if len(self.buffer) < self.cfg.batch_size:
            return {"loss": 0.0}

        states, actions, rewards, next_states, dones, indices, weights = \
            self.buffer.sample(self.cfg.batch_size)

        states_t = torch.FloatTensor(states).to(self.device)
        actions_t = torch.LongTensor(actions).to(self.device)
        rewards_t = torch.FloatTensor(rewards).to(self.device)
        next_states_t = torch.FloatTensor(next_states).to(self.device)
        dones_t = torch.FloatTensor(dones).to(self.device)
        weights_t = torch.FloatTensor(weights).to(self.device)

        q_values = self.q_net(states_t).gather(1, actions_t.unsqueeze(1)).squeeze(1)

        with torch.no_grad():
            # DDQN: select action with online, evaluate with target
            next_actions = self.q_net(next_states_t).argmax(dim=1, keepdim=True)
            next_q = self.target_net(next_states_t).gather(1, next_actions).squeeze(1)
            target = rewards_t + self.cfg.gamma * next_q * (1 - dones_t)

        td_errors = (q_values - target).detach().cpu().numpy()
        loss = (weights_t * F.mse_loss(q_values, target, reduction='none')).mean()

        self.optimizer.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(self.q_net.parameters(), 10.0)
        self.optimizer.step()

        if self.use_per:
            self.buffer.update_priorities(indices, td_errors)

        self.training_step += 1
        self._update_epsilon()
        if self.training_step % self.cfg.target_update_freq == 0:
            self.target_net.load_state_dict(self.q_net.state_dict())

        return {"loss": loss.item(), "q_mean": q_values.mean().item(), "epsilon": self.epsilon}


# ═══════════════════════════════════════════════════════════════════════════
# Dueling DQN Agent
# ═══════════════════════════════════════════════════════════════════════════

class DuelingDQNAgent(BaseAgent):
    """Dueling DQN with separate value and advantage streams."""

    def __init__(self, cfg: DRLConfig = None, use_per: bool = True):
        super().__init__("Dueling_DQN", cfg)
        self.q_net = DuelingQNetwork(self.cfg.state_dim, self.cfg.action_dim, self.cfg.hidden_dims).to(self.device)
        self.target_net = copy.deepcopy(self.q_net).to(self.device)
        self.target_net.eval()
        self.optimizer = optim.Adam(self.q_net.parameters(), lr=self.cfg.learning_rate)
        self.use_per = use_per
        if use_per:
            self.buffer = PrioritizedReplayBuffer(self.cfg.buffer_size, self.cfg.state_dim)
        else:
            self.buffer = ReplayBuffer(self.cfg.buffer_size, self.cfg.state_dim)
        self.epsilon = self.cfg.epsilon_start

    def _update_epsilon(self):
        self.epsilon = max(
            self.cfg.epsilon_end,
            self.cfg.epsilon_start - (self.cfg.epsilon_start - self.cfg.epsilon_end)
            * self.training_step / self.cfg.epsilon_decay_steps
        )

    def select_action(self, state: np.ndarray, evaluate: bool = False) -> int:
        if not evaluate and np.random.random() < self.epsilon:
            return np.random.randint(self.cfg.action_dim)
        with torch.no_grad():
            state_t = torch.FloatTensor(state).unsqueeze(0).to(self.device)
            return int(self.q_net(state_t).argmax(dim=1).item())

    def store_transition(self, state, action, reward, next_state, done):
        self.buffer.push(state, action, reward, next_state, done)

    def update(self, batch=None) -> Dict[str, float]:
        if len(self.buffer) < self.cfg.batch_size:
            return {"loss": 0.0}
        states, actions, rewards, next_states, dones, indices, weights = \
            self.buffer.sample(self.cfg.batch_size)

        s = torch.FloatTensor(states).to(self.device)
        a = torch.LongTensor(actions).to(self.device)
        r = torch.FloatTensor(rewards).to(self.device)
        ns = torch.FloatTensor(next_states).to(self.device)
        d = torch.FloatTensor(dones).to(self.device)
        w = torch.FloatTensor(weights).to(self.device)

        q = self.q_net(s).gather(1, a.unsqueeze(1)).squeeze(1)
        with torch.no_grad():
            next_actions = self.q_net(ns).argmax(dim=1, keepdim=True)
            next_q = self.target_net(ns).gather(1, next_actions).squeeze(1)
            target = r + self.cfg.gamma * next_q * (1 - d)

        td_errors = (q - target).detach().cpu().numpy()
        loss = (w * F.mse_loss(q, target, reduction='none')).mean()

        self.optimizer.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(self.q_net.parameters(), 10.0)
        self.optimizer.step()

        if self.use_per:
            self.buffer.update_priorities(indices, td_errors)

        self.training_step += 1
        self._update_epsilon()
        if self.training_step % self.cfg.target_update_freq == 0:
            self.target_net.load_state_dict(self.q_net.state_dict())

        return {"loss": loss.item(), "q_mean": q.mean().item(), "epsilon": self.epsilon}

    def save(self, path: str):
        torch.save({"q_net": self.q_net.state_dict(), "step": self.training_step}, path)

    def load(self, path: str):
        ckpt = torch.load(path, map_location=self.device, weights_only=True)
        self.q_net.load_state_dict(ckpt["q_net"])
        self.target_net.load_state_dict(ckpt["q_net"])


# ═══════════════════════════════════════════════════════════════════════════
# PPO Agent
# ═══════════════════════════════════════════════════════════════════════════

class PPOAgent(BaseAgent):
    """Proximal Policy Optimization for discrete actions."""

    def __init__(self, cfg: DRLConfig = None):
        super().__init__("PPO", cfg)
        self.actor = PolicyNetwork(self.cfg.state_dim, self.cfg.action_dim, self.cfg.hidden_dims).to(self.device)
        self.critic = ValueNetwork(self.cfg.state_dim, self.cfg.hidden_dims).to(self.device)
        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=self.cfg.learning_rate)
        self.critic_optimizer = optim.Adam(self.critic.parameters(), lr=self.cfg.learning_rate * 2)

        # Rollout storage
        self.states = []
        self.actions = []
        self.rewards = []
        self.log_probs = []
        self.dones = []
        self.values = []

    def select_action(self, state: np.ndarray, evaluate: bool = False) -> int:
        state_t = torch.FloatTensor(state).unsqueeze(0).to(self.device)
        with torch.no_grad():
            probs = self.actor(state_t)
            value = self.critic(state_t)

        if evaluate:
            return int(probs.argmax(dim=1).item())

        dist = torch.distributions.Categorical(probs)
        action = dist.sample()
        self.log_probs.append(dist.log_prob(action).item())
        self.values.append(value.item())
        return int(action.item())

    def store_transition(self, state, action, reward, next_state, done):
        self.states.append(state)
        self.actions.append(action)
        self.rewards.append(reward)
        self.dones.append(done)

    def update(self, batch=None) -> Dict[str, float]:
        if len(self.states) < self.cfg.batch_size:
            return {"policy_loss": 0.0, "value_loss": 0.0}

        states = torch.FloatTensor(np.array(self.states)).to(self.device)
        actions = torch.LongTensor(self.actions).to(self.device)
        old_log_probs = torch.FloatTensor(self.log_probs).to(self.device)

        # Compute GAE
        rewards = np.array(self.rewards)
        values = np.array(self.values + [0.0])
        dones = np.array(self.dones)

        advantages = np.zeros_like(rewards)
        gae = 0
        for t in reversed(range(len(rewards))):
            delta = rewards[t] + self.cfg.gamma * values[t + 1] * (1 - dones[t]) - values[t]
            gae = delta + self.cfg.gamma * self.cfg.gae_lambda * (1 - dones[t]) * gae
            advantages[t] = gae

        returns = advantages + values[:-1]
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
        advantages_t = torch.FloatTensor(advantages).to(self.device)
        returns_t = torch.FloatTensor(returns).to(self.device)

        total_policy_loss = 0.0
        total_value_loss = 0.0

        for _ in range(self.cfg.ppo_epochs):
            indices = np.random.permutation(len(self.states))
            for start in range(0, len(indices), self.cfg.batch_size):
                end = start + self.cfg.batch_size
                if end > len(indices):
                    break
                idx = indices[start:end]

                probs = self.actor(states[idx])
                dist = torch.distributions.Categorical(probs)
                new_log_probs = dist.log_prob(actions[idx])
                entropy = dist.entropy().mean()

                ratio = torch.exp(new_log_probs - old_log_probs[idx])
                surr1 = ratio * advantages_t[idx]
                surr2 = torch.clamp(ratio, 1 - self.cfg.ppo_clip, 1 + self.cfg.ppo_clip) * advantages_t[idx]
                policy_loss = -torch.min(surr1, surr2).mean() - self.cfg.entropy_coeff * entropy

                values_pred = self.critic(states[idx])
                value_loss = F.mse_loss(values_pred, returns_t[idx])

                self.actor_optimizer.zero_grad()
                policy_loss.backward()
                nn.utils.clip_grad_norm_(self.actor.parameters(), 0.5)
                self.actor_optimizer.step()

                self.critic_optimizer.zero_grad()
                value_loss.backward()
                nn.utils.clip_grad_norm_(self.critic.parameters(), 0.5)
                self.critic_optimizer.step()

                total_policy_loss += policy_loss.item()
                total_value_loss += value_loss.item()

        # Clear rollout
        self.states.clear()
        self.actions.clear()
        self.rewards.clear()
        self.log_probs.clear()
        self.dones.clear()
        self.values.clear()
        self.training_step += 1

        n = max(1, (self.cfg.ppo_epochs * len(indices) // self.cfg.batch_size))
        return {"policy_loss": total_policy_loss / n, "value_loss": total_value_loss / n}

    def save(self, path: str):
        torch.save({"actor": self.actor.state_dict(), "critic": self.critic.state_dict()}, path)

    def load(self, path: str):
        ckpt = torch.load(path, map_location=self.device, weights_only=True)
        self.actor.load_state_dict(ckpt["actor"])
        self.critic.load_state_dict(ckpt["critic"])


# ═══════════════════════════════════════════════════════════════════════════
# SAC Agent (Discrete)
# ═══════════════════════════════════════════════════════════════════════════

class SACAgent(BaseAgent):
    """Soft Actor-Critic adapted for discrete action spaces."""

    def __init__(self, cfg: DRLConfig = None):
        super().__init__("SAC", cfg)
        self.actor = PolicyNetwork(self.cfg.state_dim, self.cfg.action_dim, self.cfg.hidden_dims).to(self.device)
        self.q1 = QNetwork(self.cfg.state_dim, self.cfg.action_dim, self.cfg.hidden_dims).to(self.device)
        self.q2 = QNetwork(self.cfg.state_dim, self.cfg.action_dim, self.cfg.hidden_dims).to(self.device)
        self.q1_target = copy.deepcopy(self.q1).to(self.device)
        self.q2_target = copy.deepcopy(self.q2).to(self.device)

        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=self.cfg.learning_rate)
        self.q1_optimizer = optim.Adam(self.q1.parameters(), lr=self.cfg.learning_rate)
        self.q2_optimizer = optim.Adam(self.q2.parameters(), lr=self.cfg.learning_rate)

        # Auto-tune temperature
        self.target_entropy = -np.log(1.0 / self.cfg.action_dim) * 0.98
        self.log_alpha = torch.zeros(1, requires_grad=True, device=self.device)
        self.alpha_optimizer = optim.Adam([self.log_alpha], lr=self.cfg.alpha_lr)

        self.buffer = ReplayBuffer(self.cfg.buffer_size, self.cfg.state_dim)

    @property
    def alpha(self):
        return self.log_alpha.exp().item()

    def select_action(self, state: np.ndarray, evaluate: bool = False) -> int:
        state_t = torch.FloatTensor(state).unsqueeze(0).to(self.device)
        with torch.no_grad():
            probs = self.actor(state_t)
        if evaluate:
            return int(probs.argmax(dim=1).item())
        dist = torch.distributions.Categorical(probs)
        return int(dist.sample().item())

    def store_transition(self, state, action, reward, next_state, done):
        self.buffer.push(state, action, reward, next_state, done)

    def update(self, batch=None) -> Dict[str, float]:
        if len(self.buffer) < self.cfg.batch_size * 4:
            return {"q_loss": 0.0, "policy_loss": 0.0, "alpha": self.alpha}

        states, actions, rewards, next_states, dones, _, _ = \
            self.buffer.sample(self.cfg.batch_size)

        s = torch.FloatTensor(states).to(self.device)
        a = torch.LongTensor(actions).to(self.device)
        r = torch.FloatTensor(rewards).to(self.device)
        ns = torch.FloatTensor(next_states).to(self.device)
        d = torch.FloatTensor(dones).to(self.device)
        alpha = self.log_alpha.exp().detach()

        # Critic update
        with torch.no_grad():
            next_probs = self.actor(ns)
            next_log_probs = torch.log(next_probs + 1e-8)
            next_q1 = self.q1_target(ns)
            next_q2 = self.q2_target(ns)
            next_q = torch.min(next_q1, next_q2)
            next_v = (next_probs * (next_q - alpha * next_log_probs)).sum(dim=1)
            target = r + self.cfg.gamma * next_v * (1 - d)

        q1_val = self.q1(s).gather(1, a.unsqueeze(1)).squeeze(1)
        q2_val = self.q2(s).gather(1, a.unsqueeze(1)).squeeze(1)
        q1_loss = F.mse_loss(q1_val, target)
        q2_loss = F.mse_loss(q2_val, target)

        self.q1_optimizer.zero_grad()
        q1_loss.backward()
        self.q1_optimizer.step()

        self.q2_optimizer.zero_grad()
        q2_loss.backward()
        self.q2_optimizer.step()

        # Actor update
        probs = self.actor(s)
        log_probs = torch.log(probs + 1e-8)
        q1_vals = self.q1(s).detach()
        q2_vals = self.q2(s).detach()
        min_q = torch.min(q1_vals, q2_vals)
        policy_loss = (probs * (alpha * log_probs - min_q)).sum(dim=1).mean()

        self.actor_optimizer.zero_grad()
        policy_loss.backward()
        self.actor_optimizer.step()

        # Alpha update
        alpha_loss = -(self.log_alpha * (log_probs.detach() + self.target_entropy).mean()).mean()
        # Simplified: use entropy of current policy
        entropy = -(probs * log_probs).sum(dim=1).mean().detach()
        alpha_loss = -self.log_alpha * (entropy - self.target_entropy).detach()

        self.alpha_optimizer.zero_grad()
        alpha_loss.backward()
        self.alpha_optimizer.step()

        # Soft update targets
        for p, tp in zip(self.q1.parameters(), self.q1_target.parameters()):
            tp.data.copy_(self.cfg.tau * p.data + (1 - self.cfg.tau) * tp.data)
        for p, tp in zip(self.q2.parameters(), self.q2_target.parameters()):
            tp.data.copy_(self.cfg.tau * p.data + (1 - self.cfg.tau) * tp.data)

        self.training_step += 1
        return {
            "q_loss": (q1_loss.item() + q2_loss.item()) / 2,
            "policy_loss": policy_loss.item(),
            "alpha": self.alpha,
        }

    def save(self, path: str):
        torch.save({
            "actor": self.actor.state_dict(),
            "q1": self.q1.state_dict(),
            "q2": self.q2.state_dict(),
            "log_alpha": self.log_alpha.data,
        }, path)

    def load(self, path: str):
        ckpt = torch.load(path, map_location=self.device, weights_only=True)
        self.actor.load_state_dict(ckpt["actor"])
        self.q1.load_state_dict(ckpt["q1"])
        self.q2.load_state_dict(ckpt["q2"])


# ═══════════════════════════════════════════════════════════════════════════
# Baseline Algorithms
# ═══════════════════════════════════════════════════════════════════════════

class RoundRobinAgent(BaseAgent):
    def __init__(self):
        super().__init__("Round_Robin")
    def select_action(self, state=None, evaluate=False):
        return encode_action(np.array([1, 1, 1]))  # Always hold (equal split maintained)
    def update(self, batch=None): return {}
    def store_transition(self, *args): pass
    def save(self, path): pass
    def load(self, path): pass

class ProportionalFairAgent(BaseAgent):
    def __init__(self):
        super().__init__("Proportional_Fair")
        self._step = 0
    def select_action(self, state=None, evaluate=False):
        if state is None:
            return encode_action(np.array([1, 1, 1]))
        # Use throughput ratios to decide adjustments
        tput_ratios = [state[i * 7 + 2] for i in range(3)]  # throughput/target
        action = []
        for r in tput_ratios:
            if r < 0.7:
                action.append(2)  # increase
            elif r > 1.2:
                action.append(0)  # decrease
            else:
                action.append(1)  # hold
        return encode_action(np.array(action))
    def update(self, batch=None): return {}
    def store_transition(self, *args): pass
    def save(self, path): pass
    def load(self, path): pass

class PriorityStaticAgent(BaseAgent):
    """Fixed allocation: URLLC=50, eMBB=30, mMTC=20."""
    def __init__(self):
        super().__init__("Priority_Static")
        self._initialized = False
    def select_action(self, state=None, evaluate=False):
        if not self._initialized:
            self._initialized = True
            return encode_action(np.array([0, 2, 0]))  # decrease eMBB, increase URLLC
        return encode_action(np.array([1, 1, 1]))  # hold
    def update(self, batch=None): return {}
    def store_transition(self, *args): pass
    def save(self, path): pass
    def load(self, path): pass

class RandomAgent(BaseAgent):
    def __init__(self):
        super().__init__("Random")
    def select_action(self, state=None, evaluate=False):
        return np.random.randint(27)
    def update(self, batch=None): return {}
    def store_transition(self, *args): pass
    def save(self, path): pass
    def load(self, path): pass


# ═══════════════════════════════════════════════════════════════════════════
# Agent Factory
# ═══════════════════════════════════════════════════════════════════════════

def create_agent(name: str, cfg: DRLConfig = None) -> BaseAgent:
    """Factory function to create agents by name."""
    agents = {
        "dqn": lambda: DQNAgent(cfg, use_per=True),
        "ddqn": lambda: DDQNAgent(cfg, use_per=True),
        "dueling_dqn": lambda: DuelingDQNAgent(cfg, use_per=True),
        "ppo": lambda: PPOAgent(cfg),
        "sac": lambda: SACAgent(cfg),
        "round_robin": RoundRobinAgent,
        "proportional_fair": ProportionalFairAgent,
        "priority_static": PriorityStaticAgent,
        "random": RandomAgent,
    }
    return agents[name.lower()]()
