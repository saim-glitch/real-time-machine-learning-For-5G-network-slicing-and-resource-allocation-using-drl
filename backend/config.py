"""
IntelliSlice Configuration
All network, DRL, and system parameters in one place.
"""

from dataclasses import dataclass, field
from typing import Dict, List
import numpy as np


# ─────────────────────────── Network Parameters ───────────────────────────

@dataclass
class NetworkConfig:
    total_prbs: int = 100
    carrier_freq_ghz: float = 3.5
    bandwidth_mhz: float = 20.0
    subcarrier_spacing_khz: int = 30
    subcarriers_per_prb: int = 12
    tti_ms: float = 1.0
    cell_radius_m: float = 500.0
    tx_power_dbm: float = 46.0
    noise_figure_db: float = 7.0
    thermal_noise_dbm_hz: float = -174.0
    num_slots_per_frame: int = 10
    efficiency_factor: float = 0.75

    @property
    def prb_bandwidth_hz(self) -> float:
        return self.subcarrier_spacing_khz * 1e3 * self.subcarriers_per_prb

    @property
    def noise_power_dbm(self) -> float:
        bw_hz = self.prb_bandwidth_hz * self.total_prbs
        return self.thermal_noise_dbm_hz + 10 * np.log10(bw_hz) + self.noise_figure_db


# ─────────────────────────── Slice Parameters ─────────────────────────────

@dataclass
class SliceConfig:
    name: str
    slice_type: str
    num_users: int
    min_prbs: int
    max_prbs: int
    latency_budget_ms: float
    throughput_target_mbps: float
    reliability_target: float
    priority_weight: float
    traffic_model: str


SLICE_CONFIGS = [
    SliceConfig(
        name="eMBB", slice_type="eMBB", num_users=30,
        min_prbs=10, max_prbs=60, latency_budget_ms=100.0,
        throughput_target_mbps=100.0, reliability_target=0.99,
        priority_weight=0.3, traffic_model="mmpp"
    ),
    SliceConfig(
        name="URLLC", slice_type="URLLC", num_users=10,
        min_prbs=5, max_prbs=30, latency_budget_ms=1.0,
        throughput_target_mbps=10.0, reliability_target=0.99999,
        priority_weight=0.5, traffic_model="deterministic"
    ),
    SliceConfig(
        name="mMTC", slice_type="mMTC", num_users=200,
        min_prbs=5, max_prbs=40, latency_budget_ms=1000.0,
        throughput_target_mbps=20.0, reliability_target=0.95,
        priority_weight=0.2, traffic_model="beta"
    ),
]


# ─────────────────────────── DRL Parameters ───────────────────────────────

@dataclass
class DRLConfig:
    state_dim: int = 25
    num_slices: int = 3
    delta_prb: int = 5
    action_per_slice: int = 3       # {-delta, 0, +delta}
    action_dim: int = 27            # 3^3
    gamma: float = 0.99
    learning_rate: float = 1e-4
    batch_size: int = 64
    buffer_size: int = 100_000
    target_update_freq: int = 500
    epsilon_start: float = 1.0
    epsilon_end: float = 0.01
    epsilon_decay_steps: int = 10_000
    tau: float = 0.005              # Soft update
    hidden_dims: List[int] = field(default_factory=lambda: [256, 256, 128])
    # PPO specific
    ppo_clip: float = 0.2
    ppo_epochs: int = 4
    gae_lambda: float = 0.95
    entropy_coeff: float = 0.01
    # SAC specific
    alpha_lr: float = 3e-4
    # PER
    per_alpha: float = 0.6
    per_beta_start: float = 0.4
    per_beta_end: float = 1.0
    per_epsilon: float = 1e-6


# ─────────────────────────── Training Parameters ──────────────────────────

@dataclass
class TrainingConfig:
    total_episodes: int = 500
    steps_per_episode: int = 1000
    eval_interval: int = 10
    eval_episodes: int = 5
    save_best: bool = True
    log_dir: str = "data/training_logs"
    model_dir: str = "data/trained_models"
    results_dir: str = "data/results"
    seed: int = 42


# ─────────────────────────── API Parameters ───────────────────────────────

@dataclass
class APIConfig:
    host: str = "0.0.0.0"
    port: int = 8000
    ws_update_interval_ms: int = 100
    cors_origins: List[str] = field(default_factory=lambda: [
        "http://localhost:3000", "http://localhost:5173", "http://127.0.0.1:3000"
    ])


# ─────────────────────────── Traffic Parameters ───────────────────────────

@dataclass
class TrafficConfig:
    # eMBB MMPP
    embb_lambda_high: float = 50.0
    embb_lambda_low: float = 5.0
    embb_q12: float = 0.1
    embb_q21: float = 0.05
    embb_web_lambda: float = 10.0
    embb_pkt_size_mu: float = 8.5
    embb_pkt_size_sigma: float = 1.2
    # URLLC
    urllc_periodic_rate: float = 100.0
    urllc_pkt_size: int = 32
    urllc_burst_lambda: float = 5.0
    urllc_burst_size: int = 256
    # mMTC
    mmtc_beta_alpha: float = 2.0
    mmtc_beta_beta: float = 5.0
    mmtc_report_period_s: float = 10.0
    mmtc_jitter_s: float = 2.0
    mmtc_pkt_size: int = 20


# ─────────────────────────── Channel Parameters ──────────────────────────

@dataclass
class ChannelConfig:
    pathloss_a: float = 28.0
    pathloss_b: float = 22.0
    shadow_std_db: float = 8.0
    shadow_corr_dist_m: float = 50.0
    min_distance_m: float = 10.0
    cqi_sinr_thresholds: List[float] = field(default_factory=lambda: [
        -6.7, -4.7, -2.3, 0.2, 2.4, 4.3, 5.9, 8.1,
        10.3, 11.7, 14.1, 16.3, 18.7, 21.0, 23.4
    ])
