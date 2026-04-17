"""
PPO Training Module for MarketMakerEnv
======================================

This module:

- Creates a vectorized environment with multiple MarketMakerEnv instances.
- Configures and trains a PPO agent using Stable Baselines 3.
- Logs key market-making metrics (PnL, inventory).
- Saves periodic checkpoints and the final trained model.
- Provides convenience functions for training and loading agents.
"""

from pathlib import Path

from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.callbacks import CheckpointCallback, BaseCallback
import numpy as np

from src.env.market_env import MarketMakerEnv


class MetricsCallback(BaseCallback):
    

    def __init__(self, verbose=0):
        super().__init__(verbose)
        self._ep_pnls = []
        self._ep_inventories = []

    def _on_step(self):
        infos = self.locals.get("infos", [])
        dones = self.locals.get("dones", [])
        for done, info in zip(dones, infos):
            if "inventory" in info:
                self._ep_inventories.append(abs(info["inventory"]))
            if done and "realized_pnl" in info:
                self._ep_pnls.append(info["realized_pnl"])
                if self._ep_pnls:
                    self.logger.record("market/episode_pnl", self._ep_pnls[-1])
                if len(self._ep_pnls) >= 10:
                    self.logger.record(
                        "market/avg_pnl_10ep", np.mean(self._ep_pnls[-10:])
                    )
                if self._ep_inventories:
                    self.logger.record(
                        "market/avg_abs_inventory",
                        np.mean(self._ep_inventories[-200:]),
                    )
        return True


class MarketMakerTrainer:
    

    def __init__(
        self,
        env_kwargs=None,
        total_timesteps=500_000,
        learning_rate=3e-4,
        gamma=0.99,
        n_steps=2048,
        batch_size=64,
        n_envs=4,
        policy_kwargs=None,
        checkpoint_dir="checkpoints",
        seed=42,
    ):
        self.env_kwargs = env_kwargs or {}
        self.total_timesteps = total_timesteps
        self.learning_rate = learning_rate
        self.gamma = gamma
        self.n_steps = n_steps
        self.batch_size = batch_size
        self.n_envs = n_envs
        self.policy_kwargs = policy_kwargs or {"net_arch": [256, 256]}
        self.checkpoint_dir = Path(checkpoint_dir)
        self.seed = seed

        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)

    def train(self):
        vec_env = make_vec_env(
            MarketMakerEnv,
            n_envs=self.n_envs,
            env_kwargs=self.env_kwargs,
            seed=self.seed,
        )

        model = PPO(
            policy="MlpPolicy",
            env=vec_env,
            learning_rate=self.learning_rate,
            gamma=self.gamma,
            n_steps=self.n_steps,
            batch_size=self.batch_size,
            policy_kwargs=self.policy_kwargs,
            verbose=1,
            seed=self.seed,
        )

        callbacks = [
            CheckpointCallback(
                save_freq=max(10_000 // self.n_envs, 1),
                save_path=str(self.checkpoint_dir),
                name_prefix="market_maker",
            ),
            MetricsCallback(),
        ]

        model.learn(total_timesteps=self.total_timesteps, callback=callbacks)

        final_path = str(self.checkpoint_dir / "market_maker_final")
        model.save(final_path)
        print(f"\nModel saved to {final_path}.zip")
        return model, final_path + ".zip"


def train_agent(env_kwargs=None, **trainer_kwargs):
    
    trainer = MarketMakerTrainer(env_kwargs=env_kwargs, **trainer_kwargs)
    return trainer.train()


def load_agent(checkpoint_path, env=None):
    
    return PPO.load(checkpoint_path, env=env)
