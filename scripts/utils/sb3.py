import logging
import os
import sys
from typing import Optional

if sys.version_info >= (3, 8):
    from typing import Literal
else:
    from typing_extensions import Literal

from stable_baselines3.common.callbacks import BaseCallback  # type: ignore

import wandb
from wandb.sdk.lib import telemetry as wb_telemetry

logger = logging.getLogger(__name__)
import torch as th
import numpy as np

class WandbCallback(BaseCallback):
    def __init__(
        self,
        verbose: int = 0,
        model_save_path: Optional[str] = None,
        model_save_freq: int = 0,
        gradient_save_freq: int = 0,
        log: Optional[Literal["gradients", "parameters", "all"]] = "all",
        eval_env=None,
        eval_freq: int = 5000,
        n_eval_episodes: int = 5,
    ) -> None:
        super().__init__(verbose)

        if wandb.run is None:
            raise wandb.Error("You must call wandb.init() before WandbCallback()")

        # Enable SB3 telemetry
        with wb_telemetry.context() as tel:
            tel.feature.sb3 = True

        # Store callback parameters
        self.model_save_freq = model_save_freq
        self.model_save_path = model_save_path
        self.gradient_save_freq = gradient_save_freq
        self.eval_env = eval_env
        self.eval_freq = eval_freq
        self.n_eval_episodes = n_eval_episodes

        # Validate logging option
        if log not in ["gradients", "parameters", "all", None]:
            wandb.termwarn(
                "`log` must be one of `None`, 'gradients', 'parameters', or 'all'. Defaulting to 'all'."
            )
            log = "all"
        self.log = log

        # Create save directory if specified
        if self.model_save_path is not None:
            os.makedirs(self.model_save_path, exist_ok=True)
        elif self.model_save_freq > 0:
            raise ValueError("To use `model_save_freq`, you must set `model_save_path`.")

    def _init_callback(self) -> None:
        """Initialize callback: log model metadata and optionally watch gradients/parameters."""
        model_info = {"algo": type(self.model).__name__}

        # Log basic model attributes to wandb config
        for key, value in self.model.__dict__.items():
            if key not in wandb.config:
                if isinstance(value, (float, int, str)):
                    model_info[key] = value
                else:
                    model_info[key] = str(value)

        wandb.config.setdefaults(model_info)

        # Track gradients/parameters if required
        if self.gradient_save_freq > 0:
            wandb.watch(
                self.model.policy,
                log=self.log,
                log_freq=self.gradient_save_freq,
            )

    def _on_step(self) -> bool:
        """Perform actions during training step: save model, evaluate, etc."""
        if self.model_save_freq > 0 and self.model_save_path is not None:
            if self.n_calls % self.model_save_freq == 0:
                self.save_model()

        if self.eval_env is not None and self.eval_freq > 0:
            if self.n_calls % self.eval_freq == 0:
                self.evaluate()

        return True

    def evaluate(self):
        """Evaluate the current policy and log metrics to wandb."""
        print(f"Evaluating the model for {self.n_eval_episodes} episodes.")
        episode_rewards, episode_lengths, episode_successes = [], [], []

        for _ in range(self.n_eval_episodes):
            obs = self.eval_env.reset()
            done = False
            reward_sum, step_count, success = 0, 0, 0

            while not done:
                with th.no_grad():
                    action, _ = self.model.predict(obs, deterministic=True)
                obs, reward, done, info = self.eval_env.step(action)
                reward_sum += reward
                step_count += 1
                success = info.get('is_success', 0)

            episode_rewards.append(reward_sum)
            episode_lengths.append(step_count)
            episode_successes.append(success)

        # Calculate mean metrics
        avg_reward = np.mean(episode_rewards)
        avg_length = np.mean(episode_lengths)
        success_rate = np.mean(episode_successes)

        print(f"Eval results: Avg Reward: {avg_reward}, Success Rate: {success_rate}, Avg Episode Length: {avg_length}")

        # Log to wandb
        wandb.log({
            'eval/avg_reward': avg_reward,
            'eval/success_rate': success_rate,
            'eval/avg_length': avg_length
        }, step=self.num_timesteps)

    def _on_training_end(self) -> None:
        """Save the final model at the end of training."""
        if self.model_save_path is not None:
            self.save_model()

    def save_model(self) -> None:
        """Save model checkpoint and upload to wandb."""
        model_filename = f"model_{self.n_calls}.zip"
        model_path = os.path.join(self.model_save_path, model_filename)
        self.model.save(model_path)
        wandb.save(model_path, base_path=self.model_save_path)

        if self.verbose > 1:
            logger.info(f"Saved model checkpoint to {model_path}")
