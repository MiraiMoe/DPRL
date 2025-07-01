import socket
import sys
import datetime
import gym
import gym_env
import numpy as np
from stable_baselines3 import TD3, PPO, SAC
from stable_baselines3.common.noise import NormalActionNoise
from stable_baselines3.common.vec_env import SubprocVecEnv
from wandb.integration.sb3 import WandbCallback
import wandb
import ast
from configparser import ConfigParser
import torch as th
import os
import threading

from .custom_policy_sb3 import *
from .sb3 import WandbCallback

# ---------------------------- Path Setup ----------------------------

# Get the base directory of the current script
base_dir = os.path.dirname(os.path.abspath(__file__))

# Move two levels up to get the project root directory
project_root = os.path.normpath(os.path.join(base_dir, '..', '..'))

# Add project root and gym_env to sys.path for module resolution
sys.path.append(project_root)
sys.path.append(os.path.join(project_root, 'gym_env'))

# ------------------------ Environment Setup ------------------------

# Set W&B offline mode (disable auto-uploading)
os.environ["WANDB_MODE"] = "offline"

# Set global random seed for reproducibility
np.random.seed(0)
th.manual_seed(0)
th.cuda.manual_seed(0)

def make_env(cfg, rank, port, difficulty_level):
    def _init():
        # Create an AirSim client with a specific port for each environment
        env = gym.make('airsim-multi-env-v0')
        env.seed(0 + rank)  # Set seed for each environment instance to ensure diversity
        env.set_config(cfg)
        env.set_client_port(port)  # Set specific port for each environment
        env.set_obstacles(difficulty_level)
        return env
    return _init

class TrainingThread(threading.Thread):
    """
    training thread for policy model learning using MAE strategy.
    """

    def __init__(self, config):
        super(TrainingThread, self).__init__()
        print("Initializing training thread")

        # Load configuration from .ini file
        self.cfg = ConfigParser()
        self.cfg.read(config)

        # Read basic training setup
        self.project_name = self.cfg.get('options', 'env_name')
        self.use_eval = self.cfg.getboolean('options', 'use_eval')

        # make parallel environments using SubprocVecEnv
        num_envs = self.cfg.getint('options', 'num_envs')
        base_port = self.cfg.getint('options', 'base_port')  # Base port for the first environment
        self.env = SubprocVecEnv([make_env(self.cfg, i, base_port + i, 2) for i in range(num_envs)])

        # make env for evaluation
        if self.use_eval:
            self.eval_env = gym.make('airsim-multi-env-v0')
            self.eval_env.seed(10)
            self.eval_env.set_config(self.cfg)
            self.eval_env.set_client_port(41455)
            self.eval_env.set_obstacles(2)
        else:
            self.eval_env = None

        # Read hyperparameters
        self.total_timesteps = self.cfg.getint('options', 'total_timesteps')
        self.actor_feature_dim = self.cfg.getint('options', 'actor_feature_dim')
        self.critic_feature_dim = self.cfg.getint('options', 'critic_feature_dim')
        self.perception = self.cfg.get('options', 'perception')

        # Initialize wandb if enabled
        if self.cfg.getboolean('options', 'use_wandb'):
            wandb.init(
                project=self.project_name,
                notes=socket.gethostname(),
                name=self.cfg.get('wandb', 'name'),
                sync_tensorboard=True,  # auto-upload sb3's tensorboard metrics
                save_code=True,  # optional
            )

    def terminate(self):
        """
        Called when the thread is manually terminated.
        """
        print('TrainingThread terminated')

    def run(self):
        """
        Main training procedure.
        """
        print("Running training thread")

        # ------------------- Logging and Saving Paths -------------------
        now = datetime.datetime.now()
        now_string = now.strftime('%Y_%m_%d_%H_%M')
        file_path = os.path.join('logs_new', self.project_name, now_string + '_' +
                                 self.cfg.get('options', 'dynamic_name') + '_' +
                                 self.cfg.get('options', 'policy_name') + '_' +
                                 self.cfg.get('options', 'algo'))

        log_path = os.path.join(file_path, 'tb_logs')
        model_path = os.path.join(file_path, 'models')
        config_path = os.path.join(file_path, 'config')
        data_path = os.path.join(file_path, 'data')

        # Create necessary folders
        os.makedirs(log_path, exist_ok=True)
        os.makedirs(model_path, exist_ok=True)
        os.makedirs(config_path, exist_ok=True)
        os.makedirs(data_path, exist_ok=True)  # create data path to save q_map

        # Save current config file
        with open(os.path.join(config_path, 'config.ini'), 'w') as configfile:
            self.cfg.write(configfile)

        # ------------------- Policy Configuration -----------------------
        feature_num_state = self.env.get_attr('state_feature_length')[0]
        feature_num_cnn = self.cfg.getint('options', 'cnn_feature_num')
        policy_name = self.cfg.get('options', 'policy_name')
        privileged_info = self.cfg.get('options', 'privileged_info')

        # Select activation function
        activation_str = self.cfg.get('options', 'activation_function')
        activation_map = {
            'tanh': th.nn.Tanh,
            'relu': th.nn.ReLU,
            'leaky_relu': th.nn.LeakyReLU,
            'gelu': th.nn.GELU
        }
        activation_function = activation_map.get(activation_str, th.nn.ReLU)

        # Select policy network based on perception type
        policy_base = 'CnnPolicy'
        if self.perception == 'depth':
            actor_policy_used = CNN_GAP_BN
            critic_policy_used = CNN_GAP_BN
        elif self.perception == 'depth_noise':
            if privileged_info == 'no':
                actor_policy_used = Noise_Actor
                critic_policy_used = Noise_Critic
            elif privileged_info == 'noise_symmetry':
                actor_policy_used = Noise_Actor
                critic_policy_used = Noise_Critic_Symmetry
            elif privileged_info == 'noise_asymmetry':
                actor_policy_used = Noise_Actor
                critic_policy_used = Noise_Critic_Asymmetry

        # Policy keyword arguments
        policy_kwargs = dict(
            actor_features_extractor_class=actor_policy_used,
            critic_features_extractor_class=critic_policy_used,
            actor_features_extractor_kwargs=dict(
                features_dim=self.actor_feature_dim,
                state_feature_dim=feature_num_state),
            critic_features_extractor_kwargs=dict(
                features_dim=self.critic_feature_dim,
                state_feature_dim=feature_num_state),
            activation_fn=activation_function)

        # Set MLP architecture after feature extraction
        net_arch_list = ast.literal_eval(self.cfg.get("options", "net_arch"))
        policy_kwargs['net_arch'] = net_arch_list

        # --------------------- Algorithm Initialization ---------------------
        algo = self.cfg.get('options', 'algo')
        print('algo: ', algo)
        if algo == 'TD3':
            n_actions = self.env.action_space.shape[-1]
            noise_sigma = self.cfg.getfloat('TD3', 'action_noise_sigma') * np.ones(n_actions)
            action_noise = NormalActionNoise(mean=np.zeros(n_actions), sigma=noise_sigma)

            # Load existing model or initialize a new one
            if self.cfg.getboolean('options', 'load_model'):
                model = TD3.load(self.cfg.get('options', 'load_path'), self.env)
            else:
                model = TD3(
                    policy_base,
                    self.env,
                    action_noise=action_noise,
                    learning_rate=self.cfg.getfloat('TD3', 'learning_rate'),
                    gamma=self.cfg.getfloat('TD3', 'gamma'),
                    policy_kwargs=policy_kwargs,
                    learning_starts=self.cfg.getint('TD3', 'learning_starts'),
                    batch_size=self.cfg.getint('TD3', 'batch_size'),
                    train_freq=(self.cfg.getint('TD3', 'train_freq'), 'step'),
                    gradient_steps=self.cfg.getint('TD3', 'gradient_steps'),
                    buffer_size=self.cfg.getint('TD3', 'buffer_size'),
                    tensorboard_log=log_path,
                    seed=0,
                    verbose=2,
                )
        else:
            raise Exception('Invalid algo name : ', algo)

        # -------------------------- Model Training --------------------------

        print('start training model')
        self.env.model = model
        self.env.data_path = data_path

        if self.cfg.getboolean('options', 'use_wandb'):
            model.learn(
                self.total_timesteps,
                log_interval=1,
                callback=WandbCallback(
                    eval_env=self.eval_env,
                    eval_freq=10000,
                    n_eval_episodes=10,
                    model_save_freq=10000,
                    gradient_save_freq=5000,
                    model_save_path=model_path,
                    verbose=2,
                )
            )
        else:
            model.learn(self.total_timesteps)

            # ------------------------- Save Final Model -------------------------

            model.save(os.path.join(model_path, 'model_sb3'))

            print('Training finished')
            print('Model saved to: {}'.format(model_path))

