import sys
import os
import math
import pickle
from configparser import ConfigParser
from PyQt5 import QtCore
import numpy as np
from stable_baselines3 import TD3
import gym_env  # Custom AirSim gym wrapper
import gym

# ------------------------ Path Setup ------------------------

# Get base directory of current file
base_dir = os.path.dirname(os.path.abspath(__file__))

# Compute project root by moving two levels up
project_root = os.path.normpath(os.path.join(base_dir, '..', '..'))

# Append custom gym environment to Python path
sys.path.append(project_root)
sys.path.append(os.path.join(project_root, 'gym_env'))


# ---------------------- Rule-Based Policy ----------------------

def rule_based_policy(obs):
    """
    A simple linear rule-based policy for comparison (e.g., LGMD baseline).
    It transforms observation range from [-1, 1] to [0, 1],
    then applies a weighted sum and clamps the output angle.
    """
    obs = np.squeeze(obs, axis=0)
    obs = obs / 2 + 0.5

    weights = np.array([1.0, 3.0, 3.0, -3.0, -1.0, 3.0])
    action_sum = np.sum(obs * weights)

    # Clamp the action within [-40°, 40°] in radians
    action_sum = np.clip(action_sum, -math.radians(40), math.radians(40))

    return np.array([action_sum])


# ------------------------ Evaluation Thread ------------------------

class EvaluateThread(QtCore.QThread):
    def __init__(self, eval_path, config, model_file, eval_ep_num, eval_env=None, eval_dynamics=None):
        super(EvaluateThread, self).__init__()
        print("Initializing evaluation thread...")

        # Load configuration
        self.cfg = ConfigParser()
        self.cfg.read(config)

        # Override environment or dynamics name if specified
        if eval_env:
            self.cfg.set('options', 'env_name', eval_env)
        if eval_env == 'NH_center':
            self.cfg.set('environment', 'accept_radius', str(1))
        if eval_dynamics:
            self.cfg.set('options', 'dynamic_name', eval_dynamics)

        # Create and configure environment
        self.env = gym.make('airsim-env-v0')
        self.env.set_config(self.cfg)

        # Store paths and evaluation settings
        self.eval_path = eval_path
        self.model_file = model_file
        self.eval_ep_num = eval_ep_num
        self.eval_env = self.cfg.get('options', 'env_name')
        self.eval_dynamics = self.cfg.get('options', 'dynamic_name')

    def terminate(self):
        print('Evaluation terminated.')

    def run(self):
        # self.run_rule_policy()  # Optional: rule-based policy test
        return self.run_drl_model()

    def run_drl_model(self):
        """
        Evaluate the trained DRL model using TD3 over several episodes.
        Save trajectory, raw state, action, and evaluation results.
        """
        print('Starting DRL model evaluation...')
        algo = self.cfg.get('options', 'algo')

        if algo == 'TD3':
            model = TD3.load(self.model_file, env=self.env)
        else:
            raise ValueError(f'Unsupported algorithm: {algo}')

        self.env.model = model

        # Initialize tracking variables
        obs = self.env.reset()
        episode_num = 0
        reward_sum = np.array([.0])
        episode_successes, episode_crashes, step_num_list = [], [], []
        traj_list_all, action_list_all, state_list_all, obs_list_all = [], [], [], []

        traj_list, action_list, state_raw_list, obs_list = [], [], [], []

        while episode_num < self.eval_ep_num:
            unscaled_action, _ = model.predict(obs, deterministic=True)
            new_obs, reward, done, info = self.env.step(unscaled_action)

            state_raw = self.env.dynamic_model._get_state_feature_raw()
            pose = self.env.dynamic_model.get_position()

            traj_list.append(pose)
            action_list.append(unscaled_action)
            obs_list.append(obs)
            state_raw_list.append(state_raw)
            reward_sum[-1] += reward
            obs = new_obs

            if done:
                episode_num += 1
                is_success = info.get('is_success')
                is_crash = info.get('is_crash')
                print(f"Episode {episode_num} | Reward: {reward_sum[-1]:.2f} | Success: {is_success}")

                # Statistics
                episode_successes.append(float(is_success))
                episode_crashes.append(float(is_crash))
                if is_success:
                    traj_list.append(1)
                    action_list.append(1)
                    step_num_list.append(info.get('step_num'))
                elif is_crash:
                    traj_list.append(2)
                    action_list.append(2)
                else:
                    traj_list.append(3)
                    action_list.append(3)

                # Store episode data
                traj_list_all.append(traj_list)
                action_list_all.append(action_list)
                state_list_all.append(state_raw_list)
                obs_list_all.append(obs_list)

                # Reset
                traj_list, action_list, state_raw_list, obs_list = [], [], [], []
                reward_sum = np.append(reward_sum, .0)
                obs = self.env.reset()

        # Save evaluation results
        eval_folder = os.path.join(self.eval_path, f'eval_{self.eval_ep_num}_{self.eval_env}_{self.eval_dynamics}')
        os.makedirs(eval_folder, exist_ok=True)

        np.save(os.path.join(eval_folder, 'traj_eval.npy'), np.array(traj_list_all, dtype=object))
        np.save(os.path.join(eval_folder, 'action_eval.npy'), np.array(action_list_all, dtype=object))
        np.save(os.path.join(eval_folder, 'results.npy'),
                np.array([
                    reward_sum[:self.eval_ep_num].mean(),
                    np.mean(episode_successes),
                    np.mean(episode_crashes),
                    np.mean(step_num_list)
                ]))

        # Save raw state data
        try:
            with open(os.path.join(eval_folder, 'state_raw.pkl'), 'wb') as f:
                pickle.dump(state_list_all, f)
            print(f"Saved raw states to {eval_folder}/state_raw.pkl")
        except Exception as e:
            print(f"Error saving state data: {e}")

        # Print summary
        print(f"Avg Reward: {reward_sum[:self.eval_ep_num].mean():.2f} | "
              f"Success Rate: {np.mean(episode_successes):.2f} | "
              f"Crash Rate: {np.mean(episode_crashes):.2f} | "
              f"Avg Steps (Success): {np.mean(step_num_list):.2f}")

        return [
            reward_sum[:self.eval_ep_num].mean(),
            np.mean(episode_successes),
            np.mean(episode_crashes),
            np.mean(step_num_list)
        ]
