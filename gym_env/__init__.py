from importlib.metadata import entry_points
from gym.envs.registration import register

register(
    id = 'airsim-env-v0',
    entry_point = 'gym_env.envs:AirsimGymEnv'
)

register(
    id='airsim-multi-env-v0',
    entry_point='gym_env.envs:MultiEnvAirsimGymEnv'  # 指向多智能体环境的类
)
