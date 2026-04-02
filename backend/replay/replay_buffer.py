"""
Experience Replay Buffers for DRL agents.
Includes uniform replay and Prioritized Experience Replay (PER) with SumTree.
"""

import numpy as np
from typing import Tuple, Optional


class ReplayBuffer:
    """Uniform experience replay buffer."""

    def __init__(self, capacity: int, state_dim: int):
        self.capacity = capacity
        self.ptr = 0
        self.size = 0
        self.states = np.zeros((capacity, state_dim), dtype=np.float32)
        self.actions = np.zeros(capacity, dtype=np.int64)
        self.rewards = np.zeros(capacity, dtype=np.float32)
        self.next_states = np.zeros((capacity, state_dim), dtype=np.float32)
        self.dones = np.zeros(capacity, dtype=np.float32)

    def push(self, state, action, reward, next_state, done):
        self.states[self.ptr] = state
        self.actions[self.ptr] = action
        self.rewards[self.ptr] = reward
        self.next_states[self.ptr] = next_state
        self.dones[self.ptr] = float(done)
        self.ptr = (self.ptr + 1) % self.capacity
        self.size = min(self.size + 1, self.capacity)

    def sample(self, batch_size: int, rng: np.random.Generator = None):
        rng = rng or np.random.default_rng()
        indices = rng.choice(self.size, size=batch_size, replace=False)
        return (
            self.states[indices],
            self.actions[indices],
            self.rewards[indices],
            self.next_states[indices],
            self.dones[indices],
            indices,
            np.ones(batch_size, dtype=np.float32),  # uniform weights
        )

    def __len__(self):
        return self.size


class SumTree:
    """Binary sum-tree for O(log n) prioritized sampling."""

    def __init__(self, capacity: int):
        self.capacity = capacity
        self.tree = np.zeros(2 * capacity - 1, dtype=np.float64)
        self.data_ptr = 0
        self.n_entries = 0

    def _propagate(self, idx: int, change: float):
        parent = (idx - 1) // 2
        self.tree[parent] += change
        if parent != 0:
            self._propagate(parent, change)

    def _retrieve(self, idx: int, s: float) -> int:
        left = 2 * idx + 1
        right = left + 1
        if left >= len(self.tree):
            return idx
        if s <= self.tree[left]:
            return self._retrieve(left, s)
        else:
            return self._retrieve(right, s - self.tree[left])

    def total(self) -> float:
        return self.tree[0]

    def add(self, priority: float):
        idx = self.data_ptr + self.capacity - 1
        self.update(idx, priority)
        self.data_ptr = (self.data_ptr + 1) % self.capacity
        self.n_entries = min(self.n_entries + 1, self.capacity)

    def update(self, tree_idx: int, priority: float):
        change = priority - self.tree[tree_idx]
        self.tree[tree_idx] = priority
        self._propagate(tree_idx, change)

    def get(self, s: float) -> Tuple[int, float, int]:
        tree_idx = self._retrieve(0, s)
        data_idx = tree_idx - self.capacity + 1
        return tree_idx, self.tree[tree_idx], data_idx

    def min_priority(self) -> float:
        leaf_start = self.capacity - 1
        leaves = self.tree[leaf_start:leaf_start + self.n_entries]
        return float(np.min(leaves)) if self.n_entries > 0 else 1.0


class PrioritizedReplayBuffer:
    """Prioritized Experience Replay with SumTree."""

    def __init__(self, capacity: int, state_dim: int,
                 alpha: float = 0.6, beta_start: float = 0.4,
                 beta_end: float = 1.0, beta_steps: int = 100_000,
                 epsilon: float = 1e-6):
        self.capacity = capacity
        self.alpha = alpha
        self.beta = beta_start
        self.beta_start = beta_start
        self.beta_end = beta_end
        self.beta_steps = beta_steps
        self.epsilon = epsilon
        self.step_count = 0

        self.tree = SumTree(capacity)
        self.states = np.zeros((capacity, state_dim), dtype=np.float32)
        self.actions = np.zeros(capacity, dtype=np.int64)
        self.rewards = np.zeros(capacity, dtype=np.float32)
        self.next_states = np.zeros((capacity, state_dim), dtype=np.float32)
        self.dones = np.zeros(capacity, dtype=np.float32)
        self.ptr = 0
        self.size = 0
        self.max_priority = 1.0

    def push(self, state, action, reward, next_state, done):
        self.states[self.ptr] = state
        self.actions[self.ptr] = action
        self.rewards[self.ptr] = reward
        self.next_states[self.ptr] = next_state
        self.dones[self.ptr] = float(done)

        priority = self.max_priority ** self.alpha
        self.tree.add(priority)

        self.ptr = (self.ptr + 1) % self.capacity
        self.size = min(self.size + 1, self.capacity)

    def sample(self, batch_size: int, rng: np.random.Generator = None):
        self.step_count += 1
        self.beta = min(self.beta_end,
                        self.beta_start + (self.beta_end - self.beta_start) *
                        self.step_count / self.beta_steps)

        indices = np.zeros(batch_size, dtype=np.int64)
        tree_indices = np.zeros(batch_size, dtype=np.int64)
        priorities = np.zeros(batch_size, dtype=np.float64)

        segment = self.tree.total() / batch_size
        for i in range(batch_size):
            s = np.random.uniform(segment * i, segment * (i + 1))
            tree_idx, priority, data_idx = self.tree.get(s)
            data_idx = data_idx % self.size
            tree_indices[i] = tree_idx
            priorities[i] = max(priority, self.epsilon)
            indices[i] = data_idx

        # Importance sampling weights
        probs = priorities / max(self.tree.total(), self.epsilon)
        weights = (self.size * probs) ** (-self.beta)
        weights /= max(weights.max(), self.epsilon)

        return (
            self.states[indices],
            self.actions[indices],
            self.rewards[indices],
            self.next_states[indices],
            self.dones[indices],
            tree_indices,
            weights.astype(np.float32),
        )

    def update_priorities(self, tree_indices: np.ndarray, td_errors: np.ndarray):
        for idx, td in zip(tree_indices, td_errors):
            priority = (abs(td) + self.epsilon) ** self.alpha
            self.tree.update(int(idx), priority)
            self.max_priority = max(self.max_priority, abs(td) + self.epsilon)

    def __len__(self):
        return self.size
