"""
Training Orchestrator — runs episodes, evaluates, logs metrics, compares algorithms.
"""

import numpy as np
import pandas as pd
import os
import time
import json
import logging
from typing import Dict, List, Optional

import sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from config import DRLConfig, TrainingConfig
from environment.network_env import NetworkSlicingEnv
from agents.agents import create_agent, encode_action, decode_action, BaseAgent

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("trainer")


class Trainer:
    """Orchestrates DRL training, evaluation, and comparison."""

    def __init__(self, drl_cfg: DRLConfig = None, train_cfg: TrainingConfig = None):
        self.drl_cfg = drl_cfg or DRLConfig()
        self.train_cfg = train_cfg or TrainingConfig()
        os.makedirs(self.train_cfg.log_dir, exist_ok=True)
        os.makedirs(self.train_cfg.model_dir, exist_ok=True)
        os.makedirs(self.train_cfg.results_dir, exist_ok=True)
        self.training_active = False
        self.progress = {"episode": 0, "total": 0, "reward": 0.0, "status": "idle"}

    def train_agent(self, agent_name: str, callback=None) -> pd.DataFrame:
        """Train a single agent and return episode metrics."""
        logger.info(f"Training {agent_name} for {self.train_cfg.total_episodes} episodes...")
        self.training_active = True

        env = NetworkSlicingEnv({"seed": self.train_cfg.seed, "max_steps": self.train_cfg.steps_per_episode})
        agent = create_agent(agent_name, self.drl_cfg)

        records = []
        best_reward = -float("inf")

        for ep in range(1, self.train_cfg.total_episodes + 1):
            state, _ = env.reset(seed=self.train_cfg.seed + ep)
            episode_reward = 0.0
            episode_metrics = {"throughput": np.zeros(3), "latency": np.zeros(3),
                               "drop_rate": np.zeros(3), "utilization": 0.0}
            losses = []

            for step in range(self.train_cfg.steps_per_episode):
                action_idx = agent.select_action(state)
                multi_action = decode_action(action_idx)
                next_state, reward, done, truncated, info = env.step(multi_action)

                if hasattr(agent, "store_transition"):
                    agent.store_transition(state, action_idx, reward, next_state, done)

                if agent_name.lower() not in ["round_robin", "proportional_fair", "priority_static", "random"]:
                    if agent_name.lower() in ["ppo"]:
                        if done or (step + 1) % 256 == 0:
                            loss_info = agent.update()
                            if loss_info:
                                losses.append(loss_info)
                    else:
                        if step % 4 == 0:
                            loss_info = agent.update()
                            if loss_info:
                                losses.append(loss_info)

                state = next_state
                episode_reward += reward
                episode_metrics["throughput"] += np.array(info["throughput"])
                episode_metrics["latency"] += np.array(info["latency"])
                episode_metrics["drop_rate"] += np.array(info["drop_rate"])
                episode_metrics["utilization"] += info["utilization"]

                if done:
                    break

            n = self.train_cfg.steps_per_episode
            avg_loss = np.mean([l.get("loss", l.get("policy_loss", 0)) for l in losses]) if losses else 0

            record = {
                "episode": ep,
                "reward": episode_reward,
                "avg_throughput_embb": episode_metrics["throughput"][0] / n,
                "avg_throughput_urllc": episode_metrics["throughput"][1] / n,
                "avg_throughput_mmtc": episode_metrics["throughput"][2] / n,
                "avg_latency_embb": episode_metrics["latency"][0] / n,
                "avg_latency_urllc": episode_metrics["latency"][1] / n,
                "avg_latency_mmtc": episode_metrics["latency"][2] / n,
                "avg_drop_embb": episode_metrics["drop_rate"][0] / n,
                "avg_drop_urllc": episode_metrics["drop_rate"][1] / n,
                "avg_drop_mmtc": episode_metrics["drop_rate"][2] / n,
                "avg_utilization": episode_metrics["utilization"] / n,
                "loss": avg_loss,
                "epsilon": getattr(agent, "epsilon", 0),
                "alpha": getattr(agent, "alpha", 0) if hasattr(agent, "alpha") else 0,
            }
            records.append(record)

            # Save best model
            if episode_reward > best_reward and self.train_cfg.save_best:
                best_reward = episode_reward
                save_path = os.path.join(self.train_cfg.model_dir, f"{agent_name}_best.pt")
                agent.save(save_path)

            # Progress
            self.progress = {
                "episode": ep, "total": self.train_cfg.total_episodes,
                "reward": round(episode_reward, 3), "status": "training",
                "agent": agent_name, "best_reward": round(best_reward, 3),
            }

            if ep % 10 == 0:
                avg_r = np.mean([r["reward"] for r in records[-10:]])
                logger.info(f"  [{agent_name}] Ep {ep}/{self.train_cfg.total_episodes} | "
                            f"Reward: {episode_reward:.3f} | Avg10: {avg_r:.3f} | "
                            f"Loss: {avg_loss:.4f}")

            if callback:
                callback(record)

        df = pd.DataFrame(records)
        csv_path = os.path.join(self.train_cfg.log_dir, f"{agent_name}_training.csv")
        df.to_csv(csv_path, index=False)
        logger.info(f"Training log saved: {csv_path}")

        self.training_active = False
        self.progress["status"] = "complete"
        return df

    def evaluate_agent(self, agent_name: str, num_episodes: int = None) -> Dict:
        """Evaluate a trained agent and return average metrics."""
        num_episodes = num_episodes or self.train_cfg.eval_episodes
        env = NetworkSlicingEnv({"seed": self.train_cfg.seed + 9999, "max_steps": self.train_cfg.steps_per_episode})
        agent = create_agent(agent_name, self.drl_cfg)

        model_path = os.path.join(self.train_cfg.model_dir, f"{agent_name}_best.pt")
        if os.path.exists(model_path):
            agent.load(model_path)

        all_rewards = []
        all_metrics = {"throughput": np.zeros(3), "latency": np.zeros(3),
                       "drop_rate": np.zeros(3), "utilization": 0.0,
                       "sla_violations": 0, "total_steps": 0}

        for ep in range(num_episodes):
            state, _ = env.reset(seed=self.train_cfg.seed + 10000 + ep)
            ep_reward = 0.0
            for step in range(self.train_cfg.steps_per_episode):
                action_idx = agent.select_action(state, evaluate=True)
                multi_action = decode_action(action_idx)
                state, reward, done, _, info = env.step(multi_action)
                ep_reward += reward
                all_metrics["throughput"] += np.array(info["throughput"])
                all_metrics["latency"] += np.array(info["latency"])
                all_metrics["drop_rate"] += np.array(info["drop_rate"])
                all_metrics["utilization"] += info["utilization"]
                all_metrics["total_steps"] += 1
                if info["latency"][1] > 1.0:
                    all_metrics["sla_violations"] += 1
                if done:
                    break
            all_rewards.append(ep_reward)

        n = max(all_metrics["total_steps"], 1)
        return {
            "agent": agent_name,
            "avg_reward": float(np.mean(all_rewards)),
            "std_reward": float(np.std(all_rewards)),
            "avg_throughput_embb": float(all_metrics["throughput"][0] / n),
            "avg_throughput_urllc": float(all_metrics["throughput"][1] / n),
            "avg_throughput_mmtc": float(all_metrics["throughput"][2] / n),
            "avg_latency_embb": float(all_metrics["latency"][0] / n),
            "avg_latency_urllc": float(all_metrics["latency"][1] / n),
            "avg_latency_mmtc": float(all_metrics["latency"][2] / n),
            "avg_drop_embb": float(all_metrics["drop_rate"][0] / n),
            "avg_drop_urllc": float(all_metrics["drop_rate"][1] / n),
            "avg_drop_mmtc": float(all_metrics["drop_rate"][2] / n),
            "avg_utilization": float(all_metrics["utilization"] / n),
            "sla_violation_rate": float(all_metrics["sla_violations"] / n),
        }

    def compare_all(self, agents: List[str] = None, train: bool = True) -> pd.DataFrame:
        """Train and evaluate all algorithms, return comparison table."""
        agents = agents or ["dqn", "ddqn", "dueling_dqn", "ppo", "sac",
                            "round_robin", "proportional_fair", "priority_static", "random"]
        results = []
        for name in agents:
            logger.info(f"\n{'='*60}\n  Comparing: {name}\n{'='*60}")
            if train and name in ["dqn", "ddqn", "dueling_dqn", "ppo", "sac"]:
                self.train_agent(name)
            eval_result = self.evaluate_agent(name)
            results.append(eval_result)

        df = pd.DataFrame(results)
        csv_path = os.path.join(self.train_cfg.results_dir, "comparison_results.csv")
        df.to_csv(csv_path, index=False)
        logger.info(f"\nComparison results saved: {csv_path}")
        return df


# ═══════════════════════════════════════════════════════════════════════════
# Quick training script
# ═══════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--agent", default="ddqn", help="Agent to train")
    parser.add_argument("--episodes", type=int, default=100, help="Number of episodes")
    parser.add_argument("--compare", action="store_true", help="Run full comparison")
    args = parser.parse_args()

    train_cfg = TrainingConfig(total_episodes=args.episodes)
    trainer = Trainer(train_cfg=train_cfg)

    if args.compare:
        df = trainer.compare_all(train=True)
        print("\n" + df.to_string())
    else:
        df = trainer.train_agent(args.agent)
        print(f"\nFinal reward: {df['reward'].iloc[-1]:.3f}")
        print(f"Best reward:  {df['reward'].max():.3f}")
