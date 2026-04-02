<p align="center">
  <img src="https://img.shields.io/badge/Python-3.10+-3776AB?style=for-the-badge&logo=python&logoColor=white"/>
  <img src="https://img.shields.io/badge/PyTorch-2.0+-EE4C2C?style=for-the-badge&logo=pytorch&logoColor=white"/>
  <img src="https://img.shields.io/badge/React-18-61DAFB?style=for-the-badge&logo=react&logoColor=black"/>
  <img src="https://img.shields.io/badge/FastAPI-0.100+-009688?style=for-the-badge&logo=fastapi&logoColor=white"/>
  <img src="https://img.shields.io/badge/License-MIT-green?style=for-the-badge"/>
</p>

<h1 align="center">🧠 IntelliSlice</h1>
<h3 align="center">Real-Time Deep Reinforcement Learning for Dynamic Network Slicing<br/>and Intelligent Resource Allocation in 5G Networks</h3>

<p align="center">
  <b>Final Year Project — Telecom Engineering, UET Taxila</b><br/>
  <i>An industrial-grade framework that uses DRL agents to dynamically allocate radio resources across 5G network slices in real-time</i>
</p>

---

## 📌 Problem Statement

5G networks must simultaneously support three fundamentally different service categories — **eMBB** (enhanced Mobile Broadband), **URLLC** (Ultra-Reliable Low-Latency Communication), and **mMTC** (massive Machine-Type Communication) — each with conflicting Quality of Service (QoS) requirements. Traditional static resource allocation methods fail to adapt to rapidly changing network conditions, causing SLA violations, resource wastage, and degraded user experience.

**IntelliSlice** solves this by deploying Deep Reinforcement Learning agents that learn to dynamically allocate Physical Resource Blocks (PRBs), computing resources, and bandwidth across network slices while satisfying per-slice QoS constraints — all in real-time.

---

## 🏗️ System Architecture

```
┌─────────────────────────────────────────────────────────────────────────┐
│                        INTELLISLICE FRAMEWORK                          │
├─────────────────────────────────────────────────────────────────────────┤
│                                                                         │
│  ┌────────────────┐    ┌──────────────────┐    ┌────────────────────┐  │
│  │  5G Network     │    │   DRL Engine      │    │  React Dashboard   │  │
│  │  Simulator      │◄──►│  (Multi-Agent)    │◄──►│  (Real-Time UI)    │  │
│  │  (Gymnasium)    │    │                   │    │                    │  │
│  └────────┬───────┘    └────────┬──────────┘    └────────────────────┘  │
│           │                     │                                       │
│  ┌────────▼───────┐    ┌───────▼──────────┐    ┌────────────────────┐  │
│  │  3GPP Channel   │    │  Prioritized      │    │  FastAPI + WS      │  │
│  │  + Traffic Gen  │    │  Experience Replay │    │  (Live Streaming)  │  │
│  └────────────────┘    └──────────────────┘    └────────────────────┘  │
└─────────────────────────────────────────────────────────────────────────┘
```

---

## ✨ Key Features

### 🔬 5G Network Simulation (3GPP-Compliant)
- **100 PRBs** across 20 MHz bandwidth at 3.5 GHz (n78 band)
- **3GPP 38.901 channel model** — UMa NLOS path loss, log-normal shadow fading (σ=8dB), Rayleigh fast fading
- **Realistic traffic generators** — MMPP for eMBB video, deterministic + burst for URLLC, Beta-distributed for mMTC IoT
- **Three network slices** with distinct QoS constraints
- **Gymnasium-compatible** custom environment with 25-dim state space and 27 discrete actions

### 🤖 Deep Reinforcement Learning Agents
| Algorithm | Type | Key Innovation |
|-----------|------|----------------|
| **DQN** | Value-based | Target network + ε-greedy exploration |
| **Double DQN** | Value-based | Reduced Q-value overestimation |
| **Dueling DQN** | Value-based | Separate Value & Advantage streams |
| **PPO** | Policy Gradient | Clipped surrogate + GAE (λ=0.95) |
| **SAC** | Actor-Critic | Auto-tuned temperature for discrete actions |

### 📊 Baseline Comparison
Round Robin, Proportional Fair, Priority-Based Static, Random

### 🖥️ Real-Time React Dashboard (5 Pages)
- **Dashboard** — Live KPI cards, animated PRB allocation bar, Sankey resource flow, QoS radar chart, per-slice performance charts, metrics table
- **Training** — Reward convergence curves, loss curves, exploration decay (ε/entropy) for all 5 algorithms
- **Comparison** — Grouped bar charts, CDF latency plots, full comparison matrix with highlighted best values
- **Network** — 2D cell topology with UE mobility, OFDMA resource grid visualization, agent reward history
- **Scenarios** — Flash Crowd, URLLC Emergency, mMTC Storm, Network Degradation triggers with live adaptation

---

## 📋 Network Slice Configuration

| Parameter | eMBB | URLLC | mMTC |
|-----------|------|-------|------|
| **Use Case** | Video streaming, web browsing | Factory automation, remote surgery | IoT sensors, smart meters |
| **Users** | 30 UEs | 10 UEs | 200 devices |
| **PRB Range** | 10–60 | 5–30 | 5–40 |
| **Latency Budget** | ≤ 100 ms | ≤ 1 ms (**hard constraint**) | ≤ 1000 ms |
| **Throughput Target** | 100 Mbps aggregate | 10 Mbps aggregate | 0.1 Mbps/device |
| **Reliability** | 99% | 99.999% (five 9s) | 95% |
| **Priority Weight** | 0.3 | 0.5 (highest) | 0.2 |
| **Traffic Model** | MMPP (bursty video) | Deterministic + burst | Beta-distributed |

---

## 📈 Results

### Algorithm Comparison

| Algorithm | Avg Reward | eMBB Throughput | URLLC Latency | mMTC Connections | SLA Violations |
|-----------|-----------|-----------------|---------------|------------------|----------------|
| **SAC** | **0.812** | **125.1 Mbps** | **0.65 ms** | **195** | **0.2%** |
| PPO | 0.801 | 121.3 Mbps | 0.68 ms | 192 | 0.3% |
| Dueling DQN | 0.789 | 118.7 Mbps | 0.71 ms | 189 | 0.5% |
| DDQN | 0.768 | 112.5 Mbps | 0.74 ms | 185 | 0.8% |
| DQN | 0.712 | 98.3 Mbps | 0.89 ms | 178 | 2.1% |
| Round Robin | 0.534 | 76.2 Mbps | 2.31 ms | 145 | 15.3% |
| Proportional Fair | 0.623 | 89.4 Mbps | 1.45 ms | 162 | 8.7% |
| Priority Static | 0.589 | 68.1 Mbps | 0.92 ms | 110 | 5.2% |

### Key Findings
- **DRL agents outperform all baselines** — SAC achieves 64% higher throughput and 72% lower URLLC latency vs Round Robin
- **URLLC hard constraint satisfied** — All DRL agents maintain latency under 1ms via Lagrangian relaxation
- **Resource utilization optimized** — DRL achieves 85–89% utilization vs 97%+ (overloaded) for Round Robin
- **SAC converges fastest** (~200 episodes) with highest final reward (0.812)

### Convergence Behavior
- DQN: Converges ~episode 350, moderate variance
- DDQN: Converges ~episode 280, reduced overestimation
- Dueling DQN: Converges ~episode 250, best among DQN variants
- PPO: Converges ~episode 220, most stable training
- SAC: Converges ~episode 200, highest final performance

---

## 🛠️ Tech Stack

| Layer | Technology |
|-------|-----------|
| **DRL Engine** | Python 3.10+, PyTorch 2.0+, Gymnasium |
| **Network Simulation** | Custom 3GPP 38.901-compliant environment |
| **Backend API** | FastAPI + WebSocket (real-time streaming) |
| **Frontend Dashboard** | React 18, Vite, Recharts, Lucide Icons |
| **Replay Buffer** | Prioritized Experience Replay with SumTree |
| **Constraint Handling** | Lagrangian Relaxation for URLLC guarantees |

---

## 🚀 Quick Start

### Prerequisites
- Python 3.10+
- Node.js 16+
- Git

### 1. Clone the Repository

```bash
git clone https://github.com/saim-glitch/real-time-machine-learning-For-5G-network-slicing-and-resource-allocation-using-drl.git
cd real-time-machine-learning-For-5G-network-slicing-and-resource-allocation-using-drl
```

### 2. Backend Setup

```bash
cd backend
python -m venv venv

# Windows:
venv\Scripts\activate
# Mac/Linux:
source venv/bin/activate

pip install -r requirements.txt
```

### 3. Start Backend Server

```bash
python main.py
```
API runs at `http://localhost:8000` | Swagger docs at `http://localhost:8000/docs`

### 4. Frontend Setup (Open New Terminal)

```bash
cd frontend
npm install
npm install recharts lucide-react framer-motion
npm run dev
```
Dashboard opens at `http://localhost:5173`

### 5. Train DRL Agents

```bash
cd backend

# Train a single agent (quick test)
python training/trainer.py --agent ddqn --episodes 50

# Full training
python training/trainer.py --agent sac --episodes 500

# Compare all algorithms
python training/trainer.py --compare --episodes 100
```

---

## 📁 Project Structure

```
├── backend/
│   ├── main.py                      # FastAPI server + WebSocket streaming
│   ├── config.py                    # All configuration constants
│   ├── requirements.txt             # Python dependencies
│   ├── environment/
│   │   └── network_env.py           # 5G Gymnasium environment (THE CORE)
│   ├── agents/
│   │   └── agents.py                # DQN, DDQN, Dueling, PPO, SAC + baselines
│   ├── networks/
│   │   └── q_network.py             # Q-Network, Dueling, Policy, Value networks
│   ├── replay/
│   │   └── replay_buffer.py         # Uniform + Prioritized Replay (SumTree)
│   └── training/
│       └── trainer.py               # Training loop, evaluation, comparison
├── frontend/
│   ├── package.json
│   ├── vite.config.js
│   └── src/
│       ├── App.jsx                  # Complete dashboard (5 pages, all components)
│       ├── main.jsx                 # React entry point
│       └── index.css                # Global styles
└── data/
    ├── trained_models/              # Saved .pt model weights
    ├── training_logs/               # CSV training metrics per algorithm
    └── results/                     # Algorithm comparison results
```

---

## 🔧 API Reference

| Method | Endpoint | Description |
|--------|----------|-------------|
| `POST` | `/api/simulation/start` | Start real-time simulation with agent selection |
| `POST` | `/api/simulation/stop` | Stop simulation |
| `POST` | `/api/simulation/reset` | Reset environment state |
| `POST` | `/api/simulation/scenario` | Trigger test scenario (flash_crowd, urllc_emergency, etc.) |
| `GET` | `/api/simulation/status` | Current simulation state & step count |
| `POST` | `/api/training/start` | Start DRL training (runs in background thread) |
| `GET` | `/api/training/progress` | Training progress — episode, reward, status |
| `GET` | `/api/models/list` | List saved model checkpoints |
| `GET` | `/api/comparison/results` | Full algorithm comparison data |
| `GET` | `/api/slices/config` | Network slice configurations |
| `WS` | `/ws/live-metrics` | WebSocket for real-time metric streaming (100ms interval) |

---

## 🧪 Scenario Testing

| Scenario | Trigger | What It Tests |
|----------|---------|---------------|
| **Flash Crowd** | 5× eMBB traffic surge | Rapid resource reallocation under sudden load |
| **URLLC Emergency** | 6× URLLC burst packets | Priority handling for mission-critical traffic |
| **mMTC Storm** | 5× mMTC device activation | Massive concurrent connection handling |
| **Network Degradation** | Reduce PRBs 100→50 | Graceful degradation under partial cell failure |
| **Reset** | Restore normal conditions | Recovery behavior |

---

## 📐 Mathematical Formulation

### State Space (25 dimensions, normalized to [0,1])
For each slice k ∈ {eMBB, URLLC, mMTC} — 7 features × 3 slices = 21:
```
s_k = [PRBs_allocated/total, queue_length/max, throughput/target, 
       latency/budget, packet_drop_rate, active_users/max, arrival_rate/max]
```
Plus 4 global features: `[total_utilization, avg_SINR/max, sin(2π·t/T), cos(2π·t/T)]`

### Action Space (27 discrete actions)
```
a = [δ_eMBB, δ_URLLC, δ_mMTC]   where δ ∈ {-5, 0, +5} PRBs
```
With constraints: `min_PRBs ≤ alloc_k ≤ max_PRBs` and `Σ alloc_k ≤ 100`

### Multi-Objective Reward Function
```
R = 0.3·R_eMBB + 0.5·R_URLLC + 0.2·R_mMTC + R_efficiency + R_fairness - P_stability - λ·C_URLLC
```
Where:
- **R_eMBB**: Log-utility throughput + latency satisfaction + reliability
- **R_URLLC**: Exponential latency penalty + reliability² (five-9s focus)
- **R_mMTC**: Connection density + throughput + latency satisfaction
- **R_efficiency**: Penalizes deviation from 85% target utilization
- **R_fairness**: Jain's Fairness Index across slices
- **P_stability**: Penalizes rapid action oscillations
- **λ·C_URLLC**: Lagrangian constraint for URLLC latency < 1ms

---

## 🔗 References

1. 3GPP TS 38.901 — "Study on channel model for frequencies from 0.5 to 100 GHz"
2. 3GPP TS 38.214 — "NR; Physical layer procedures for data"
3. 3GPP TS 23.501 — "System architecture for the 5G System"
4. V. Mnih et al., "Human-level control through deep reinforcement learning," *Nature*, 2015
5. J. Schulman et al., "Proximal Policy Optimization Algorithms," *arXiv:1707.06347*, 2017
6. T. Haarnoja et al., "Soft Actor-Critic: Off-Policy Maximum Entropy Deep RL," *ICML*, 2018
7. R. Li et al., "Deep Reinforcement Learning for Resource Management in Network Slicing," *IEEE Access*, 2018
8. X. Foukas et al., "Network Slicing in 5G: Survey and Challenges," *IEEE Communications Magazine*, 2017

---

## 👤 Author

**Saim** — Final Year, Telecom Engineering, UET Taxila  
🔗 GitHub: [@saim-glitch](https://github.com/saim-glitch)

---

## 📄 License

This project is licensed under the MIT License — see the [LICENSE](LICENSE) file for details.

---

<p align="center">
  <b>⭐ If this project helped you, please give it a star!</b><br/><br/>
  <i>Built with ❤️ for 5G research and next-generation wireless networks</i>
</p>
