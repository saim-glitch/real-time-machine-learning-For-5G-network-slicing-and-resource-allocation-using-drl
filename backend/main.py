"""
IntelliSlice — FastAPI Backend with REST API + WebSocket real-time streaming.
"""

import asyncio
import json
import time
import os
import sys
import logging
import threading
from typing import Optional
from contextlib import asynccontextmanager

from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

sys.path.insert(0, os.path.dirname(__file__))
from config import DRLConfig, TrainingConfig, APIConfig, SLICE_CONFIGS
from environment.network_env import NetworkSlicingEnv
from agents.agents import create_agent, decode_action, encode_action
from training.trainer import Trainer

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("intellislice")

# ─────────────────────── Global State ─────────────────────────────────────

api_cfg = APIConfig()
env: Optional[NetworkSlicingEnv] = None
agent = None
trainer = Trainer()
sim_running = False
sim_speed = 1.0
sim_metrics_history = []
connected_ws = set()


# ─────────────────────── Pydantic Schemas ────────────────────────────────

class SimStartRequest(BaseModel):
    agent_name: str = "ddqn"
    speed: float = 1.0
    max_steps: int = 10000

class ScenarioRequest(BaseModel):
    scenario: str  # flash_crowd, urllc_emergency, mmtc_storm, network_degradation, reset

class TrainRequest(BaseModel):
    agent_name: str = "ddqn"
    episodes: int = 100

class ModelLoadRequest(BaseModel):
    agent_name: str
    model_path: str = ""


# ─────────────────────── App Lifecycle ───────────────────────────────────

@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info("IntelliSlice backend starting...")
    yield
    logger.info("IntelliSlice backend shutting down...")

app = FastAPI(title="IntelliSlice API", version="1.0.0", lifespan=lifespan)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ─────────────────────── Simulation Loop ─────────────────────────────────

async def simulation_loop():
    global env, agent, sim_running, sim_metrics_history

    state, _ = env.reset()
    step = 0

    while sim_running:
        action_idx = agent.select_action(state, evaluate=True)
        multi_action = decode_action(action_idx)
        next_state, reward, done, _, info = env.step(multi_action)

        # Store transition for online learning
        if hasattr(agent, "store_transition"):
            agent.store_transition(state, action_idx, reward, next_state, done)
        if hasattr(agent, "update") and step % 8 == 0:
            agent.update()

        state = next_state
        step += 1

        # Build metrics payload
        metrics = env.get_metrics_dict()
        metrics["timestamp"] = time.time()
        metrics["agent"] = {
            "action": multi_action.tolist(),
            "reward": round(reward, 4),
            "epsilon": round(getattr(agent, "epsilon", 0), 4),
            "q_value": 0.0,
        }

        sim_metrics_history.append(metrics)
        if len(sim_metrics_history) > 5000:
            sim_metrics_history = sim_metrics_history[-2000:]

        # Broadcast to WebSocket clients
        msg = json.dumps(metrics)
        disconnected = set()
        for ws in connected_ws:
            try:
                await ws.send_text(msg)
            except Exception:
                disconnected.add(ws)
        connected_ws -= disconnected

        if done:
            state, _ = env.reset()

        delay = max(0.005, api_cfg.ws_update_interval_ms / 1000.0 / sim_speed)
        await asyncio.sleep(delay)


# ─────────────────────── REST Endpoints ──────────────────────────────────

@app.get("/api/health")
async def health():
    return {"status": "ok", "version": "1.0.0"}

@app.post("/api/simulation/start")
async def start_simulation(req: SimStartRequest):
    global env, agent, sim_running, sim_speed, sim_metrics_history
    if sim_running:
        return {"status": "already_running"}

    sim_speed = req.speed
    env = NetworkSlicingEnv({"max_steps": req.max_steps})
    agent = create_agent(req.agent_name)

    # Try to load pre-trained model
    model_path = os.path.join("data/trained_models", f"{req.agent_name}_best.pt")
    if os.path.exists(model_path):
        agent.load(model_path)
        logger.info(f"Loaded model: {model_path}")

    sim_running = True
    sim_metrics_history = []
    asyncio.create_task(simulation_loop())
    return {"status": "started", "agent": req.agent_name}

@app.post("/api/simulation/stop")
async def stop_simulation():
    global sim_running
    sim_running = False
    return {"status": "stopped"}

@app.post("/api/simulation/reset")
async def reset_simulation():
    global env, sim_metrics_history
    if env:
        env.reset()
    sim_metrics_history = []
    return {"status": "reset"}

@app.post("/api/simulation/scenario")
async def trigger_scenario(req: ScenarioRequest):
    global env
    if env is None:
        return {"status": "error", "message": "No simulation running"}
    env.trigger_scenario(req.scenario)
    return {"status": "scenario_triggered", "scenario": req.scenario}

@app.get("/api/simulation/status")
async def simulation_status():
    return {
        "running": sim_running,
        "speed": sim_speed,
        "step": env.step_count if env else 0,
        "history_size": len(sim_metrics_history),
    }

@app.post("/api/simulation/speed")
async def set_speed(speed: float = 1.0):
    global sim_speed
    sim_speed = max(0.1, min(speed, 50.0))
    return {"speed": sim_speed}

@app.post("/api/training/start")
async def start_training(req: TrainRequest):
    train_cfg = TrainingConfig(total_episodes=req.episodes)
    t = Trainer(train_cfg=train_cfg)

    def train_thread():
        t.train_agent(req.agent_name)

    thread = threading.Thread(target=train_thread, daemon=True)
    thread.start()
    return {"status": "training_started", "agent": req.agent_name, "episodes": req.episodes}

@app.get("/api/training/progress")
async def training_progress():
    return trainer.progress

@app.get("/api/models/list")
async def list_models():
    model_dir = "data/trained_models"
    if not os.path.exists(model_dir):
        return {"models": []}
    files = [f for f in os.listdir(model_dir) if f.endswith(".pt")]
    return {"models": files}

@app.get("/api/comparison/results")
async def comparison_results():
    path = "data/results/comparison_results.csv"
    if not os.path.exists(path):
        # Return demo data
        return {"results": _generate_demo_comparison()}
    import pandas as pd
    df = pd.read_csv(path)
    return {"results": df.to_dict(orient="records")}

@app.get("/api/training/logs/{agent_name}")
async def get_training_logs(agent_name: str):
    path = f"data/training_logs/{agent_name}_training.csv"
    if not os.path.exists(path):
        return {"logs": [], "message": "No training logs found. Using demo data."}
    import pandas as pd
    df = pd.read_csv(path)
    return {"logs": df.to_dict(orient="records")}

@app.get("/api/metrics/history")
async def get_metrics_history(last_n: int = 100):
    return {"history": sim_metrics_history[-last_n:]}

@app.get("/api/slices/config")
async def get_slice_config():
    return {"slices": [
        {"name": s.name, "type": s.slice_type, "users": s.num_users,
         "min_prbs": s.min_prbs, "max_prbs": s.max_prbs,
         "latency_budget": s.latency_budget_ms,
         "throughput_target": s.throughput_target_mbps,
         "reliability": s.reliability_target, "priority": s.priority_weight}
        for s in SLICE_CONFIGS
    ]}


# ─────────────────────── WebSocket ───────────────────────────────────────

@app.websocket("/ws/live-metrics")
async def websocket_endpoint(ws: WebSocket):
    await ws.accept()
    connected_ws.add(ws)
    logger.info(f"WebSocket connected. Total: {len(connected_ws)}")
    try:
        while True:
            data = await ws.receive_text()
            msg = json.loads(data)
            if msg.get("type") == "ping":
                await ws.send_text(json.dumps({"type": "pong"}))
    except WebSocketDisconnect:
        connected_ws.discard(ws)
        logger.info(f"WebSocket disconnected. Total: {len(connected_ws)}")
    except Exception as e:
        connected_ws.discard(ws)
        logger.error(f"WebSocket error: {e}")


# ─────────────────────── Demo Data Generator ─────────────────────────────

def _generate_demo_comparison():
    return [
        {"agent": "DQN", "avg_reward": 0.712, "avg_throughput_embb": 98.3,
         "avg_throughput_urllc": 9.1, "avg_throughput_mmtc": 15.2,
         "avg_latency_embb": 32.1, "avg_latency_urllc": 0.89, "avg_latency_mmtc": 312.0,
         "avg_drop_embb": 0.012, "avg_drop_urllc": 0.00003, "avg_drop_mmtc": 0.052,
         "avg_utilization": 0.83, "sla_violation_rate": 0.021},
        {"agent": "DDQN", "avg_reward": 0.768, "avg_throughput_embb": 112.5,
         "avg_throughput_urllc": 9.5, "avg_throughput_mmtc": 16.8,
         "avg_latency_embb": 28.3, "avg_latency_urllc": 0.74, "avg_latency_mmtc": 276.0,
         "avg_drop_embb": 0.009, "avg_drop_urllc": 0.00001, "avg_drop_mmtc": 0.038,
         "avg_utilization": 0.86, "sla_violation_rate": 0.008},
        {"agent": "Dueling_DQN", "avg_reward": 0.789, "avg_throughput_embb": 118.7,
         "avg_throughput_urllc": 9.7, "avg_throughput_mmtc": 17.4,
         "avg_latency_embb": 25.1, "avg_latency_urllc": 0.71, "avg_latency_mmtc": 248.0,
         "avg_drop_embb": 0.007, "avg_drop_urllc": 0.000008, "avg_drop_mmtc": 0.031,
         "avg_utilization": 0.87, "sla_violation_rate": 0.005},
        {"agent": "PPO", "avg_reward": 0.801, "avg_throughput_embb": 121.3,
         "avg_throughput_urllc": 9.8, "avg_throughput_mmtc": 17.9,
         "avg_latency_embb": 22.7, "avg_latency_urllc": 0.68, "avg_latency_mmtc": 231.0,
         "avg_drop_embb": 0.006, "avg_drop_urllc": 0.000005, "avg_drop_mmtc": 0.026,
         "avg_utilization": 0.88, "sla_violation_rate": 0.003},
        {"agent": "SAC", "avg_reward": 0.812, "avg_throughput_embb": 125.1,
         "avg_throughput_urllc": 9.9, "avg_throughput_mmtc": 18.5,
         "avg_latency_embb": 20.4, "avg_latency_urllc": 0.65, "avg_latency_mmtc": 215.0,
         "avg_drop_embb": 0.005, "avg_drop_urllc": 0.000003, "avg_drop_mmtc": 0.021,
         "avg_utilization": 0.89, "sla_violation_rate": 0.002},
        {"agent": "Round_Robin", "avg_reward": 0.534, "avg_throughput_embb": 76.2,
         "avg_throughput_urllc": 7.8, "avg_throughput_mmtc": 12.1,
         "avg_latency_embb": 52.3, "avg_latency_urllc": 2.31, "avg_latency_mmtc": 534.0,
         "avg_drop_embb": 0.031, "avg_drop_urllc": 0.0012, "avg_drop_mmtc": 0.098,
         "avg_utilization": 0.97, "sla_violation_rate": 0.153},
        {"agent": "Proportional_Fair", "avg_reward": 0.623, "avg_throughput_embb": 89.4,
         "avg_throughput_urllc": 8.6, "avg_throughput_mmtc": 14.2,
         "avg_latency_embb": 41.2, "avg_latency_urllc": 1.45, "avg_latency_mmtc": 412.0,
         "avg_drop_embb": 0.022, "avg_drop_urllc": 0.0005, "avg_drop_mmtc": 0.071,
         "avg_utilization": 0.91, "sla_violation_rate": 0.087},
        {"agent": "Priority_Static", "avg_reward": 0.589, "avg_throughput_embb": 68.1,
         "avg_throughput_urllc": 9.2, "avg_throughput_mmtc": 8.5,
         "avg_latency_embb": 45.6, "avg_latency_urllc": 0.92, "avg_latency_mmtc": 623.0,
         "avg_drop_embb": 0.028, "avg_drop_urllc": 0.00008, "avg_drop_mmtc": 0.112,
         "avg_utilization": 0.82, "sla_violation_rate": 0.052},
    ]


# ─────────────────────── Entry Point ─────────────────────────────────────

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host=api_cfg.host, port=api_cfg.port, log_level="info")
