"""
NetworkSlicingEnv — Gymnasium-compatible 5G Network Slicing Environment
3GPP-compliant channel, traffic, and resource allocation models.
"""

import gymnasium as gym
import numpy as np
from typing import Dict, Tuple, Optional, Any
from math import log, exp, log10, pi, sin, cos
from dataclasses import dataclass

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from config import NetworkConfig, SliceConfig, SLICE_CONFIGS, ChannelConfig, TrafficConfig


# ═══════════════════════════════════════════════════════════════════════════
# Channel Model (3GPP 38.901)
# ═══════════════════════════════════════════════════════════════════════════

class ChannelModel:
    """3GPP 38.901 Urban Macro NLOS channel model."""

    def __init__(self, cfg: ChannelConfig, net_cfg: NetworkConfig, rng: np.random.Generator):
        self.cfg = cfg
        self.net = net_cfg
        self.rng = rng
        self.shadow_map: Dict[int, float] = {}

    def path_loss(self, distance_m: float) -> float:
        d = max(distance_m, self.cfg.min_distance_m)
        return (self.cfg.pathloss_a +
                self.cfg.pathloss_b * log10(d) +
                20.0 * log10(self.net.carrier_freq_ghz))

    def shadow_fading(self, user_id: int) -> float:
        if user_id not in self.shadow_map:
            self.shadow_map[user_id] = self.rng.normal(0, self.cfg.shadow_std_db)
        else:
            innovation = self.rng.normal(0, self.cfg.shadow_std_db * 0.1)
            self.shadow_map[user_id] = 0.95 * self.shadow_map[user_id] + 0.05 * innovation
        return self.shadow_map[user_id]

    def fast_fading(self) -> float:
        h_real = self.rng.normal(0, 1.0 / (2 ** 0.5))
        h_imag = self.rng.normal(0, 1.0 / (2 ** 0.5))
        return h_real ** 2 + h_imag ** 2  # Rayleigh

    def compute_sinr(self, distance_m: float, user_id: int, num_prbs: int) -> float:
        pl = self.path_loss(distance_m)
        sf = self.shadow_fading(user_id)
        ff_db = 10.0 * log10(max(self.fast_fading(), 1e-10))
        rx_power_dbm = self.net.tx_power_dbm - pl - sf + ff_db
        bw_hz = self.net.prb_bandwidth_hz * max(num_prbs, 1)
        noise_dbm = self.net.thermal_noise_dbm_hz + 10 * log10(bw_hz) + self.net.noise_figure_db
        interference_dbm = noise_dbm + self.rng.uniform(-3, 3)
        noise_plus_interference = 10 ** (noise_dbm / 10) + 10 ** (interference_dbm / 10) * 0.1
        npi_dbm = 10 * log10(max(noise_plus_interference, 1e-30))
        sinr_db = rx_power_dbm - npi_dbm
        return np.clip(sinr_db, -10.0, 40.0)

    def sinr_to_cqi(self, sinr_db: float) -> int:
        cqi = 1
        for i, thr in enumerate(ChannelConfig().cqi_sinr_thresholds):
            if sinr_db >= thr:
                cqi = i + 1
        return min(cqi, 15)

    def spectral_efficiency(self, sinr_db: float) -> float:
        sinr_linear = 10 ** (sinr_db / 10)
        return self.net.efficiency_factor * log(1 + sinr_linear) / log(2)

    def reset(self):
        self.shadow_map.clear()


# ═══════════════════════════════════════════════════════════════════════════
# Traffic Generators
# ═══════════════════════════════════════════════════════════════════════════

class TrafficGenerator:
    """Per-slice traffic generation with 3GPP-realistic models."""

    def __init__(self, tcfg: TrafficConfig, rng: np.random.Generator):
        self.cfg = tcfg
        self.rng = rng
        self.mmpp_state = 0  # 0=low, 1=high
        self.mmtc_timers: Optional[np.ndarray] = None

    def reset(self, num_mmtc_devices: int = 200):
        self.mmpp_state = 0
        self.mmtc_timers = self.rng.uniform(0, self.cfg.mmtc_report_period_s, num_mmtc_devices)

    def generate_embb(self, num_users: int) -> Tuple[float, float]:
        if self.mmpp_state == 0:
            if self.rng.random() < self.cfg.embb_q12:
                self.mmpp_state = 1
            lam = self.cfg.embb_lambda_low
        else:
            if self.rng.random() < self.cfg.embb_q21:
                self.mmpp_state = 0
            lam = self.cfg.embb_lambda_high

        video_pkts = self.rng.poisson(lam * 0.001)  # per TTI (1ms)
        web_pkts = self.rng.poisson(self.cfg.embb_web_lambda * 0.001)
        total_pkts = (video_pkts + web_pkts) * num_users
        pkt_sizes = np.exp(self.rng.normal(self.cfg.embb_pkt_size_mu, self.cfg.embb_pkt_size_sigma,
                                            max(total_pkts, 1)))
        total_bytes = float(np.sum(pkt_sizes[:total_pkts])) if total_pkts > 0 else 0.0
        total_bits = total_bytes * 8
        arrival_rate = (video_pkts + web_pkts) * num_users
        return total_bits, float(arrival_rate)

    def generate_urllc(self, num_users: int) -> Tuple[float, float]:
        periodic_pkts = int(self.cfg.urllc_periodic_rate * 0.001) * num_users
        burst_events = self.rng.poisson(self.cfg.urllc_burst_lambda * 0.001 * num_users)
        total_bytes = (periodic_pkts * self.cfg.urllc_pkt_size +
                       burst_events * self.cfg.urllc_burst_size)
        total_bits = total_bytes * 8
        arrival_rate = periodic_pkts + burst_events
        return float(total_bits), float(arrival_rate)

    def generate_mmtc(self, num_devices: int) -> Tuple[float, float]:
        if self.mmtc_timers is None:
            self.reset(num_devices)
        self.mmtc_timers -= 0.001  # 1ms TTI
        reporting = np.sum(self.mmtc_timers <= 0)
        mask = self.mmtc_timers <= 0
        jitter = self.rng.uniform(
            self.cfg.mmtc_report_period_s - self.cfg.mmtc_jitter_s,
            self.cfg.mmtc_report_period_s + self.cfg.mmtc_jitter_s,
            int(np.sum(mask))
        )
        self.mmtc_timers[mask] = jitter
        total_bits = float(reporting * self.cfg.mmtc_pkt_size * 8)
        return total_bits, float(reporting)


# ═══════════════════════════════════════════════════════════════════════════
# User Equipment
# ═══════════════════════════════════════════════════════════════════════════

@dataclass
class UE:
    ue_id: int
    slice_type: str
    x: float
    y: float
    speed: float  # m/s
    direction: float  # radians

    def distance_to_bs(self) -> float:
        return max((self.x ** 2 + self.y ** 2) ** 0.5, 10.0)

    def move(self, rng: np.random.Generator, cell_radius: float, dt_s: float = 0.001):
        if rng.random() < 0.01:
            self.direction = rng.uniform(0, 2 * pi)
        self.x += self.speed * cos(self.direction) * dt_s
        self.y += self.speed * sin(self.direction) * dt_s
        dist = self.distance_to_bs()
        if dist > cell_radius:
            self.direction += pi
            self.x = (cell_radius * 0.95) * cos(self.direction)
            self.y = (cell_radius * 0.95) * sin(self.direction)


# ═══════════════════════════════════════════════════════════════════════════
# Main Environment
# ═══════════════════════════════════════════════════════════════════════════

class NetworkSlicingEnv(gym.Env):
    """
    5G Network Slicing Environment for DRL-based resource allocation.
    
    State:  25-dim normalized vector (7 per slice + 4 global)
    Action: MultiDiscrete [3,3,3] → 27 actions (PRB adjustments)
    Reward: Multi-objective with Lagrangian URLLC constraints
    """

    metadata = {"render_modes": ["human", "json"]}

    def __init__(self, config: Optional[Dict] = None):
        super().__init__()
        cfg = config or {}
        self.net_cfg = NetworkConfig(**{k: v for k, v in cfg.items() if hasattr(NetworkConfig, k)})
        self.ch_cfg = ChannelConfig()
        self.tr_cfg = TrafficConfig()
        self.slice_configs = SLICE_CONFIGS
        self.num_slices = len(self.slice_configs)
        self.delta_prb = cfg.get("delta_prb", 5)
        self.max_steps = cfg.get("max_steps", 1000)
        self.max_queue = cfg.get("max_queue", 500)

        # Spaces
        self.observation_space = gym.spaces.Box(
            low=0.0, high=1.0, shape=(7 * self.num_slices + 4,), dtype=np.float32
        )
        self.action_space = gym.spaces.MultiDiscrete([3] * self.num_slices)

        # RNG
        self._rng = np.random.default_rng(cfg.get("seed", 42))

        # Components
        self.channel = ChannelModel(self.ch_cfg, self.net_cfg, self._rng)
        self.traffic = TrafficGenerator(self.tr_cfg, self._rng)

        # State variables
        self.allocation = np.zeros(self.num_slices, dtype=np.int32)
        self.prev_action = np.ones(self.num_slices, dtype=np.int32)
        self.queues = np.zeros(self.num_slices, dtype=np.float64)
        self.step_count = 0
        self.ues: list[UE] = []
        self.steps_per_day = cfg.get("steps_per_day", 86_400_000)

        # Running metrics (exponential moving average)
        self._ema_alpha = 0.05
        self.throughput = np.zeros(self.num_slices)
        self.latency = np.zeros(self.num_slices)
        self.drop_rate = np.zeros(self.num_slices)
        self.active_users = np.zeros(self.num_slices)
        self.arrival_rate = np.zeros(self.num_slices)
        self.connected_devices = 0.0

        # Lagrange multipliers for URLLC constraints
        self.lagrange = {"latency": 0.5, "reliability": 0.5}

        # Scenario modifiers
        self.traffic_multiplier = np.ones(self.num_slices)
        self.prb_limit = self.net_cfg.total_prbs

    # ─────────────────────── Gymnasium Interface ──────────────────────────

    def reset(self, seed=None, options=None) -> Tuple[np.ndarray, Dict]:
        super().reset(seed=seed)
        if seed is not None:
            self._rng = np.random.default_rng(seed)
            self.channel = ChannelModel(self.ch_cfg, self.net_cfg, self._rng)
            self.traffic = TrafficGenerator(self.tr_cfg, self._rng)

        base = self.net_cfg.total_prbs // self.num_slices
        self.allocation = np.array([base] * self.num_slices, dtype=np.int32)
        self.allocation[-1] += self.net_cfg.total_prbs - self.allocation.sum()
        self.prev_action = np.ones(self.num_slices, dtype=np.int32)
        self.queues = np.zeros(self.num_slices, dtype=np.float64)
        self.step_count = 0

        self.throughput = np.zeros(self.num_slices)
        self.latency = np.full(self.num_slices, 5.0)
        self.drop_rate = np.zeros(self.num_slices)
        self.active_users = np.array([sc.num_users * 0.8 for sc in self.slice_configs])
        self.arrival_rate = np.zeros(self.num_slices)
        self.connected_devices = 160.0
        self.lagrange = {"latency": 0.5, "reliability": 0.5}
        self.traffic_multiplier = np.ones(self.num_slices)
        self.prb_limit = self.net_cfg.total_prbs

        self.channel.reset()
        self.traffic.reset(self.slice_configs[2].num_users)
        self._spawn_ues()

        return self._get_obs(), self._get_info()

    def step(self, action: np.ndarray) -> Tuple[np.ndarray, float, bool, bool, Dict]:
        self.step_count += 1
        action = np.asarray(action, dtype=np.int32)

        # 1. Apply action
        self._apply_action(action)

        # 2. Move UEs
        for ue in self.ues:
            ue.move(self._rng, self.net_cfg.cell_radius_m)

        # 3. Generate traffic per slice
        traffic_bits = np.zeros(self.num_slices)
        arrivals = np.zeros(self.num_slices)

        bits, arr = self.traffic.generate_embb(self.slice_configs[0].num_users)
        traffic_bits[0] = bits * self.traffic_multiplier[0]
        arrivals[0] = arr * self.traffic_multiplier[0]

        bits, arr = self.traffic.generate_urllc(self.slice_configs[1].num_users)
        traffic_bits[1] = bits * self.traffic_multiplier[1]
        arrivals[1] = arr * self.traffic_multiplier[1]

        bits, arr = self.traffic.generate_mmtc(self.slice_configs[2].num_users)
        traffic_bits[2] = bits * self.traffic_multiplier[2]
        arrivals[2] = arr * self.traffic_multiplier[2]

        # 4. Compute per-slice capacity and metrics
        inst_throughput = np.zeros(self.num_slices)
        inst_latency = np.zeros(self.num_slices)
        inst_drop = np.zeros(self.num_slices)
        inst_active = np.zeros(self.num_slices)

        for k, sc in enumerate(self.slice_configs):
            slice_ues = [u for u in self.ues if u.slice_type == sc.slice_type]
            if not slice_ues:
                continue

            # Average spectral efficiency for this slice's users
            avg_se = 0.0
            for ue in slice_ues:
                sinr = self.channel.compute_sinr(ue.distance_to_bs(), ue.ue_id, self.allocation[k])
                avg_se += self.channel.spectral_efficiency(sinr)
            avg_se /= len(slice_ues)

            # Capacity in bits per TTI
            capacity_bps = avg_se * self.net_cfg.prb_bandwidth_hz * self.allocation[k]
            capacity_bits_per_tti = capacity_bps * 0.001  # 1ms TTI

            # Queue dynamics
            self.queues[k] += traffic_bits[k]
            served = min(self.queues[k], capacity_bits_per_tti)
            dropped = max(0, self.queues[k] - self.max_queue * 8 * 1500)
            self.queues[k] = max(0, self.queues[k] - served - dropped)

            inst_throughput[k] = served / 0.001 / 1e6  # Mbps
            queue_delay = (self.queues[k] / max(capacity_bits_per_tti, 1)) * self.net_cfg.tti_ms
            proc_delay = 0.1 if sc.slice_type == "URLLC" else 0.5
            inst_latency[k] = queue_delay + proc_delay + self._rng.exponential(0.1)
            inst_drop[k] = dropped / max(traffic_bits[k] + self.queues[k], 1)
            inst_active[k] = min(len(slice_ues), sc.num_users) * min(1.0, served / max(traffic_bits[k], 1))

        # mMTC connected devices
        inst_connected = self.slice_configs[2].num_users * (1 - inst_drop[2]) * \
                         min(1.0, self.allocation[2] / 15.0)

        # 5. EMA smoothing
        a = self._ema_alpha
        self.throughput = (1 - a) * self.throughput + a * inst_throughput
        self.latency = (1 - a) * self.latency + a * inst_latency
        self.drop_rate = (1 - a) * self.drop_rate + a * inst_drop
        self.active_users = (1 - a) * self.active_users + a * inst_active
        self.arrival_rate = (1 - a) * self.arrival_rate + a * arrivals
        self.connected_devices = (1 - a) * self.connected_devices + a * inst_connected

        # 6. Compute reward
        reward = self._compute_reward(action)
        self.prev_action = action.copy()

        # 7. Done?
        terminated = self.step_count >= self.max_steps
        return self._get_obs(), reward, terminated, False, self._get_info()

    # ─────────────────────── Internal Methods ─────────────────────────────

    def _spawn_ues(self):
        self.ues = []
        uid = 0
        for sc in self.slice_configs:
            n = sc.num_users if sc.slice_type != "mMTC" else min(sc.num_users, 50)
            for _ in range(n):
                r = self._rng.uniform(self.ch_cfg.min_distance_m, self.net_cfg.cell_radius_m)
                theta = self._rng.uniform(0, 2 * pi)
                speed = {"eMBB": 1.5, "URLLC": 0.5, "mMTC": 0.0}[sc.slice_type]
                self.ues.append(UE(
                    ue_id=uid, slice_type=sc.slice_type,
                    x=r * cos(theta), y=r * sin(theta),
                    speed=speed, direction=self._rng.uniform(0, 2 * pi)
                ))
                uid += 1

    def _apply_action(self, action: np.ndarray):
        deltas = (action.astype(np.int32) - 1) * self.delta_prb
        new_alloc = self.allocation + deltas
        for k, sc in enumerate(self.slice_configs):
            new_alloc[k] = np.clip(new_alloc[k], sc.min_prbs, sc.max_prbs)

        total = new_alloc.sum()
        if total > self.prb_limit:
            excess = total - self.prb_limit
            priorities = [sc.priority_weight for sc in self.slice_configs]
            order = np.argsort(priorities)
            for k in order:
                reduce = min(excess, new_alloc[k] - self.slice_configs[k].min_prbs)
                new_alloc[k] -= reduce
                excess -= reduce
                if excess <= 0:
                    break

        self.allocation = new_alloc.astype(np.int32)

    def _compute_reward(self, action: np.ndarray) -> float:
        m = {
            "throughput_embb": self.throughput[0],
            "throughput_urllc": self.throughput[1],
            "throughput_mmtc": self.throughput[2],
            "latency_embb": self.latency[0],
            "latency_urllc": self.latency[1],
            "latency_mmtc": self.latency[2],
            "drop_rate_embb": self.drop_rate[0],
            "drop_rate_urllc": self.drop_rate[1],
            "drop_rate_mmtc": self.drop_rate[2],
            "connected_devices": self.connected_devices,
            "total_prb_used": float(self.allocation.sum()),
        }

        # eMBB reward
        R_embb = (0.6 * log(1 + m["throughput_embb"] / 100.0) +
                  0.2 * max(0, 1 - m["latency_embb"] / 100.0) +
                  0.2 * (1 - m["drop_rate_embb"]))

        # URLLC reward
        R_urllc = (0.1 * min(m["throughput_urllc"] / 10.0, 1.0) +
                   0.5 * exp(-m["latency_urllc"] / 1.0) +
                   0.4 * (1 - m["drop_rate_urllc"]) ** 2)

        # mMTC reward
        R_mmtc = (0.2 * min(m["throughput_mmtc"] / 20.0, 1.0) +
                  0.2 * max(0, 1 - m["latency_mmtc"] / 1000.0) +
                  0.6 * min(m["connected_devices"] / 200.0, 1.0))

        reward = 0.3 * R_embb + 0.5 * R_urllc + 0.2 * R_mmtc

        # Resource efficiency
        util = m["total_prb_used"] / self.net_cfg.total_prbs
        reward += 0.1 * (1 - abs(util - 0.85) / 0.85)

        # Jain's Fairness Index
        tp = [max(self.throughput[i], 1e-8) for i in range(self.num_slices)]
        jfi = sum(tp) ** 2 / (self.num_slices * sum(t ** 2 for t in tp))
        reward += 0.05 * jfi

        # Stability penalty
        action_diff = float(np.sum(np.abs(action.astype(int) - self.prev_action.astype(int))))
        reward -= 0.02 * action_diff

        # Lagrangian URLLC constraints
        if m["latency_urllc"] > 1.0:
            violation = m["latency_urllc"] - 1.0
            reward -= self.lagrange["latency"] * violation
            self.lagrange["latency"] = max(0, self.lagrange["latency"] + 0.01 * violation)

        if m["drop_rate_urllc"] > 1e-5:
            violation = m["drop_rate_urllc"] - 1e-5
            reward -= self.lagrange["reliability"] * violation
            self.lagrange["reliability"] = max(0, self.lagrange["reliability"] + 0.01 * violation)

        return float(np.clip(reward, -2.0, 2.0))

    def _get_obs(self) -> np.ndarray:
        obs = []
        for k, sc in enumerate(self.slice_configs):
            obs.extend([
                self.allocation[k] / self.net_cfg.total_prbs,
                np.clip(self.queues[k] / (self.max_queue * 8 * 1500), 0, 1),
                np.clip(self.throughput[k] / max(sc.throughput_target_mbps, 1), 0, 2),
                np.clip(self.latency[k] / max(sc.latency_budget_ms, 0.1), 0, 2),
                np.clip(self.drop_rate[k], 0, 1),
                np.clip(self.active_users[k] / max(sc.num_users, 1), 0, 1),
                np.clip(self.arrival_rate[k] / 100.0, 0, 1),
            ])
        # Global
        obs.append(self.allocation.sum() / self.net_cfg.total_prbs)
        avg_sinr = np.mean([self.channel.compute_sinr(ue.distance_to_bs(), ue.ue_id, 10)
                            for ue in self.ues[:10]]) if self.ues else 10.0
        obs.append(np.clip(avg_sinr / 40.0, 0, 1))
        t_norm = self.step_count / self.steps_per_day
        obs.append(0.5 + 0.5 * sin(2 * pi * t_norm))
        obs.append(0.5 + 0.5 * cos(2 * pi * t_norm))

        return np.array(obs, dtype=np.float32)

    def _get_info(self) -> Dict[str, Any]:
        return {
            "step": self.step_count,
            "allocation": self.allocation.tolist(),
            "throughput": self.throughput.tolist(),
            "latency": self.latency.tolist(),
            "drop_rate": self.drop_rate.tolist(),
            "active_users": self.active_users.tolist(),
            "connected_devices": float(self.connected_devices),
            "utilization": float(self.allocation.sum() / self.net_cfg.total_prbs),
            "fairness": float(sum(self.throughput) ** 2 /
                             (self.num_slices * max(sum(t ** 2 for t in self.throughput), 1e-8))),
            "lagrange": dict(self.lagrange),
        }

    # ─────────────────────── Scenario Injection ───────────────────────────

    def trigger_scenario(self, scenario: str):
        """Inject test scenarios for robustness evaluation."""
        if scenario == "flash_crowd":
            self.traffic_multiplier[0] = 5.0
        elif scenario == "urllc_emergency":
            self.traffic_multiplier[1] = 10.0
        elif scenario == "mmtc_storm":
            self.traffic_multiplier[2] = 5.0
        elif scenario == "network_degradation":
            self.prb_limit = 50
        elif scenario == "reset":
            self.traffic_multiplier = np.ones(self.num_slices)
            self.prb_limit = self.net_cfg.total_prbs

    def get_metrics_dict(self) -> Dict[str, Any]:
        """Return current metrics formatted for WebSocket streaming."""
        return {
            "step": self.step_count,
            "allocation": {
                "embb": int(self.allocation[0]),
                "urllc": int(self.allocation[1]),
                "mmtc": int(self.allocation[2]),
            },
            "metrics": {
                "embb": {
                    "throughput": round(float(self.throughput[0]), 2),
                    "latency": round(float(self.latency[0]), 2),
                    "drop_rate": round(float(self.drop_rate[0]), 5),
                    "active_users": int(self.active_users[0]),
                },
                "urllc": {
                    "throughput": round(float(self.throughput[1]), 2),
                    "latency": round(float(self.latency[1]), 4),
                    "drop_rate": round(float(self.drop_rate[1]), 6),
                    "active_users": int(self.active_users[1]),
                },
                "mmtc": {
                    "throughput": round(float(self.throughput[2]), 2),
                    "latency": round(float(self.latency[2]), 2),
                    "drop_rate": round(float(self.drop_rate[2]), 4),
                    "connected_devices": int(self.connected_devices),
                },
            },
            "global": {
                "utilization": round(float(self.allocation.sum() / self.net_cfg.total_prbs), 3),
                "fairness_index": round(float(
                    sum(self.throughput) ** 2 /
                    (self.num_slices * max(sum(t ** 2 for t in self.throughput), 1e-8))
                ), 3),
                "sla_compliance": round(1.0 - np.mean(self.drop_rate), 4),
            },
        }
