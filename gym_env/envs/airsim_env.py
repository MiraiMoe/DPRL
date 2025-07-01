import sys
import gym
from gym import spaces
import airsim
from configparser import NoOptionError
import torch as th
import numpy as np
import math
import cv2
from collections import deque
import time
import random
import msgpackrpc
from .dynamics.multirotor_fusion import MultirotorDynamicsFuse
from PyQt5 import QtCore
from PyQt5.QtCore import pyqtSignal
from scipy.spatial.transform import Rotation


np.random.seed(0)
th.manual_seed(0)
th.cuda.manual_seed(0)


class AirsimGymEnv(gym.Env, QtCore.QThread):
    # ! ---------------------initialize pyqt-------------------------------------
    # PyQt signals for real-time visualization
    action_signal = pyqtSignal(int, np.ndarray)
    state_signal = pyqtSignal(int, np.ndarray)
    attitude_signal = pyqtSignal(int, np.ndarray, np.ndarray)
    component_signal = pyqtSignal(int, list)
    reward_signal = pyqtSignal(int, float, float)
    pose_signal = pyqtSignal(np.ndarray, np.ndarray, np.ndarray, np.ndarray)

    # ! ---------------------initialization-------------------------------------
    def __init__(self) -> None:
        super().__init__()

        # Set NumPy and PyTorch print options for better readability
        np.set_printoptions(formatter={'float': '{: 4.2f}'.format}, suppress=True)
        th.set_printoptions(profile="short", sci_mode=False, linewidth=1000)
        print("init airsim-gym-env.")

        # Initialize position tracking
        self.previous_position = [0, 0, 0]
        self.reward_components = []
        self.min_distance_to_obstacles = 5

        # Buffers to determine successful hover state
        self.success_flag = False
        self.flag_buffer = deque(maxlen=20)
        self.trajectory_list = []

        # Initialize action stacks for reward computation
        self.action_stack = np.zeros((3, 4), dtype=float)

        # Metrics for performance and curriculum learning
        self.success_rate = 0
        self.total_episodes = 0
        self.successful_episodes = 0
        self.difficulty_level = 1
        self.difficulty_levels = {
            1: 8,
            2: 6,
            3: 4,
            4: 3
        }
        self.success_buffer = deque(maxlen=100)

        # Parameters for obstacle spawning
        self.inner_radius = 10
        self.outer_radius = 50
        self.cylinder_radius = 2.5
        self.tree_types = [f"tree0{i}" for i in range(1, 4)]
        self.obstacle_type = "Shape_Cube"
        self.existing_positions = []
        self.obstacle_names = []

    def set_config(self, cfg):
        """
        Load configuration and initialize the environment.
        """
        # Read config options for environment, dynamics, and perception
        self.cfg = cfg
        self.env_name = cfg.get('options', 'env_name')
        self.dynamic_name = cfg.get('options', 'dynamic_name')
        self.perception_type = cfg.get('options', 'perception')
        self.privileged_info = cfg.get('options', 'privileged_info')

        print('Environment: ', self.env_name, "Dynamics: ", self.dynamic_name,
              'Perception: ', self.perception_type)

        # Initialize dynamic model
        if self.dynamic_name == 'MultirotorFuse':
            self.dynamic_model = MultirotorDynamicsFuse(cfg)
        else:
            raise Exception("Invalid dynamic_name!", self.dynamic_name)

        # Course learning and obstacle generation options
        self.use_cl = cfg.getboolean('options', 'use_course_learning')
        self.generate_obstacles = cfg.getboolean('options', 'generate_obstacles')

        # Configure environment-specific parameters
        if self.env_name == 'Hover':
            start_position = [0, 0, 5]
            boundary = 80
            self.goal_distance = 2
            self.obstacle_range = [[0, 30], [50, 70]]
            self.work_space_x = [start_position[0] - boundary, start_position[0] + boundary]
            self.work_space_y = [start_position[1] - boundary, start_position[1] + boundary]
            self.work_space_z = [0.2, 15]
            self.max_episode_steps = 100
            self.dynamic_model.set_start(obstacle_range=self.obstacle_range, work_space_x=self.work_space_x,
                                         work_space_y=self.work_space_y)
            self.dynamic_model.set_goal(goal_distance=self.goal_distance)
            self.fly_range = 20
            self.hover_radius = 0.1

        elif self.env_name == 'Nav':
            start_position = [0, 0, 5]
            boundary = 80
            self.goal_distance = 65
            self.obstacle_range = [[20, 60]]
            self.work_space_x = [start_position[0] - boundary, start_position[0] + boundary]
            self.work_space_y = [start_position[1] - boundary, start_position[1] + boundary]
            self.work_space_z = [1, 15]
            self.max_episode_steps = 500
            self.dynamic_model.set_start(obstacle_range=self.obstacle_range, work_space_x=self.work_space_x,
                                         work_space_y=self.work_space_y)
            self.dynamic_model.set_goal(goal_distance=self.goal_distance)
            self.fly_range = boundary

        else:
            raise Exception("Invalid env_name!", self.env_name)

        # Set state-related attributes
        self.client = self.dynamic_model.client
        self.state_feature_length = self.dynamic_model.state_feature_length
        self.cnn_feature_length = self.cfg.getint('options', 'cnn_feature_num')

        # Initialize episode statistics
        self.episode_num = 0
        self.total_step = 0
        self.step_num = 0
        self.cumulated_episode_reward = 0
        self.previous_distance_from_des_point = 0
        self.previous_direction = 0

        # Load physical/environmental parameters from config
        self.crash_distance = cfg.getfloat('multirotor_new', 'crash_distance')
        self.accept_radius = cfg.getfloat('multirotor_new', 'accept_radius')
        self.max_depth_meters = cfg.getint('environment', 'max_depth_meters')
        self.screen_height = cfg.getint('environment', 'screen_height')
        self.screen_width = cfg.getint('environment', 'screen_width')

        # Override difficulty if curriculum learning is not used
        if not self.use_cl:
            self.difficulty_level = 4

        # Generate obstacles if enabled
        if self.generate_obstacles:
            if self.env_name == 'Hover':
                self.obstacle_positions = self.generate_obstacle_positions(self.obstacle_range, min_distance=12,
                                                                           num_obstacles=30)
                self.set_all_obstacles()
            elif self.env_name == 'Nav':
                self.obstacle_positions = self.generate_obstacle_positions(self.obstacle_range, min_distance=10,
                                                                           num_obstacles=50)
                self.set_all_obstacles()

        # Configure noise models
        self.gaussian_std = cfg.getfloat('noise', 'gaussian_std')
        self.salt_pepper_prob = cfg.getfloat('noise', 'salt_pepper_prob')
        self.motion_blur_kernel_size = cfg.getint('noise', 'motion_blur_kernel_size')
        self.state_feature_noise_std = cfg.getfloat('noise', 'state_feature_noise_std')

        # Set action and observation spaces
        self.action_space = self.dynamic_model.action_space

        if self.perception_type == 'depth_noise':
            self.observation_space = spaces.Box(low=0, high=255,
                                                shape=(self.screen_height, self.screen_width, 3),
                                                dtype=np.uint8)
        elif self.perception_type == 'depth':
            self.observation_space = spaces.Box(low=0, high=255,
                                                shape=(self.screen_height, self.screen_width, 2),
                                                dtype=np.uint8)

        # Load reward type if specified
        try:
            self.reward_type = cfg.get('options', 'reward_type')
            print('Reward type: ', self.reward_type)
        except NoOptionError:
            self.reward_type = None

    # ! ---------------------key function-------------------------------------
    def reset(self):
        """
        Reset the environment for a new episode.
        """
        # Reset UAV state using dynamic model
        self.dynamic_model.set_start(self.obstacle_range, self.work_space_x, self.work_space_y)
        self.dynamic_model.set_goal(self.goal_distance)
        self.dynamic_model.reset()

        # Reset episode counters and flags
        self.episode_num += 1
        self.step_num = 0
        self.cumulated_episode_reward = 0
        self.dynamic_model.goal_distance = self.dynamic_model.get_distance_to_goal_2d()
        self.previous_distance_from_des_point = self.dynamic_model.goal_distance
        self.previous_direction = 0
        self.trajectory_list = []

        # Return initial observation
        obs = self.get_obs()
        return obs

    def step(self, action):
        """
        Execute one environment step using the given action.
        """
        # Update the action history buffer with the latest action
        self.action_stack = np.roll(self.action_stack, shift=1, axis=0)
        self.action_stack[0] = action

        # Apply action to the UAV
        self.dynamic_model.set_action(action)

        # Record UAV position for trajectory tracking
        position_ue4 = self.dynamic_model.get_position()
        self.trajectory_list.append(position_ue4)

        # Get the updated observation
        obs = self.get_obs()

        # Determine task status depending on environment name
        if self.env_name == 'Hover':
            reached_goal = self.is_reached_goal()
            self.flag_buffer.append(reached_goal)
            self.success_flag = len(self.flag_buffer) == self.flag_buffer.maxlen and all(self.flag_buffer)
            done = self.is_done()
            info = {
                'is_success': self.success_flag,
                'is_crash': self.is_crashed(),
                'is_not_in_workspace': self.is_not_inside_fly_range(),
                'step_num': self.step_num
            }
        elif self.env_name == 'Nav':
            done = self.is_done()
            info = {
                'is_success': self.is_in_desired_pose(),
                'is_crash': self.is_crashed(),
                'is_not_in_workspace': self.is_not_inside_fly_range(),
                'step_num': self.step_num
            }

        # Handle success/failure statistics and curriculum adjustment
        if done:
            print(info)
            self.success_buffer.append(1 if self.success_flag else 0)
            if self.success_flag:
                self.successful_episodes += 1
            self.total_episodes += 1
            self.update_success_rate()
            success_threshold = 0.1 * self.difficulty_level
            if self.success_rate > success_threshold:
                self.increase_difficulty()

        # Compute reward using selected reward function
        if self.reward_type == 'reward_hover':
            reward, reward_list = self.compute_reward_hover(done, self.action_stack)
        elif self.reward_type == 'reward_distance':
            reward, reward_list = self.compute_reward_distance(done, self.action_stack)
        elif self.reward_type == 'reward_doge':
            reward, reward_list = self.compute_reward_doge(done, self.action_stack)

        # Accumulate reward and send signal (e.g., for visualization or logging)
        self.cumulated_episode_reward += reward
        self.set_pyqt_signal_multirotor(action, reward, reward_list)

        # Update counters
        self.step_num += 1
        self.total_step += 1

        return obs, reward, done, info

    # ! ---------------------curriculum related-------------------------------------
    def update_success_rate(self):
        """
        Update the success rate based on recent episodes stored in the buffer.
        """
        if len(self.success_buffer) > 0:
            self.success_rate = sum(self.success_buffer) / len(self.success_buffer)
        else:
            self.success_rate = 0

    def increase_difficulty(self):
        """
        Gradually increase environment difficulty by adjusting parameters like hover radius.
        """
        if self.difficulty_level < 5:
            print(f"Increasing difficulty to level {self.difficulty_level + 1}")
            self.difficulty_level += 1
            if self.use_cl:
                self.hover_radius -= 0.01

    # ! ---------------------generate obstacles-------------------------------------
    def generate_obstacle_positions(self, obstacle_range, min_distance=10, num_obstacles=30):
        """
        Generate obstacle positions within two concentric ring areas, ensuring a minimum distance between obstacles.

        Parameters:
        - obstacle_range: list of two [min_radius, max_radius] pairs defining two rings
        - min_distance: minimum distance between obstacles
        - num_obstacles: total number of obstacles to generate
        """
        obstacle_positions = []
        if len(obstacle_range) == 2:
            while len(obstacle_positions) < num_obstacles:
                chosen_ring = random.choice([0, 1])
                radius = random.uniform(obstacle_range[chosen_ring][0], obstacle_range[chosen_ring][1])
                angle = random.uniform(0, 2 * math.pi)
                x, y, z = radius * math.cos(angle), radius * math.sin(angle), 2

                # Check distance with existing obstacles
                valid_position = all(
                    math.sqrt((x - pos.x_val) ** 2 + (y - pos.y_val) ** 2) >= min_distance
                    for pos in obstacle_positions
                )
                if valid_position:
                    obstacle_positions.append(airsim.Vector3r(x, y, z))
        else:
            while len(obstacle_positions) < num_obstacles:
                radius = random.uniform(obstacle_range[0][0], obstacle_range[0][1])
                angle = random.uniform(0, 2 * math.pi)
                x, y, z = radius * math.cos(angle), radius * math.sin(angle), 2
                valid_position = all(
                    math.sqrt((x - pos.x_val) ** 2 + (y - pos.y_val) ** 2) >= min_distance
                    for pos in obstacle_positions
                )
                if valid_position:
                    obstacle_positions.append(airsim.Vector3r(x, y, z))

        self.obstacle_positions = obstacle_positions
        return obstacle_positions

    def set_all_obstacles(self):
        """
        Spawn all obstacles in the simulation environment based on generated positions.
        """
        obstacle_positions = self.obstacle_positions
        i = 0
        while i < len(obstacle_positions):
            for _ in range(10):
                if i >= len(obstacle_positions):
                    break
                position = obstacle_positions[i]
                pose = airsim.Pose(position, airsim.Quaternionr(0, 0, 0, 1))

                # Randomly select obstacle type with higher chance for 'tree'
                obstacle_type = random.choices(
                    ["Shape_Cylinder", "Shape_Cube", "tree"],
                    weights=[0.3, 0.3, 0.4],
                    k=1
                )[0]

                if obstacle_type == "Shape_Cylinder":
                    obstacle_name = f"{obstacle_type}_Obstacle_{i + 1}"
                    scale = airsim.Vector3r(3, 3, 12)
                    segmentation_id = 10
                elif obstacle_type == "Shape_Cube":
                    obstacle_name = f"{obstacle_type}_Obstacle_{i + 1}"
                    scale = airsim.Vector3r(3, 3, 12)
                    segmentation_id = 15
                elif obstacle_type == "tree":
                    selected_tree = random.choice(self.tree_types)
                    obstacle_name = f"{selected_tree}_Obstacle_{i + 1}"
                    scale = airsim.Vector3r(1, 1, random.uniform(1.2, 1.8))
                    segmentation_id = 20
                else:
                    print(f"Unknown obstacle type: {obstacle_type}")
                    continue

                self.obstacle_names.append(obstacle_name)

                try:
                    self.dynamic_model.client.simSpawnObject(
                        obstacle_name,
                        selected_tree if obstacle_type == "tree" else obstacle_type,
                        pose,
                        scale
                    )
                    self.dynamic_model.client.simSetSegmentationObjectID(obstacle_name, segmentation_id, True)
                    print(f"Spawned {obstacle_name} at {position} with Segmentation ID {segmentation_id}")
                except Exception as e:
                    print(f"Failed to spawn {obstacle_name}: {e}")
                i += 1

            time.sleep(1.0)  # Wait before placing the next batch

    def delete_all_obstacles(self):
        """
        Delete all spawned obstacles from the simulation.
        """
        for obstacle_id in self.obstacle_names:
            self.dynamic_model.client.simDestroyObject(obstacle_id)

    def get_min_distance_to_obstacles(self):
        """
        Calculate the minimum 2D distance from the drone to the surface of all cylindrical obstacles.

        Returns:
            min_distance (float): The smallest horizontal (XY-plane) distance from the drone
                                  to the surface of any obstacle (non-negative).
        """
        # Get the current position of the drone (x, y, z)
        drone_position = self.dynamic_model.get_position()
        drone_x, drone_y, drone_z = drone_position

        # Initialize minimum distance as infinity
        min_distance = float('inf')

        # Iterate through all obstacles
        for obstacle in self.obstacle_positions:
            # Compute 2D Euclidean distance to the center of the obstacle
            distance = math.sqrt((drone_x - obstacle.x_val) ** 2 + (drone_y - obstacle.y_val) ** 2)

            # Subtract the obstacle's radius to get distance to the surface (not center)
            distance_to_surface = max(0, distance - self.cylinder_radius)

            # Update the minimum distance if this one is smaller
            if distance_to_surface < min_distance:
                min_distance = distance_to_surface

        return min_distance

    # ! ---------------------get obs-------------------------------------
    def get_obs(self):
        """
        Select observation mode based on perception type.
        Returns processed observation including depth and state features.
        """
        if self.perception_type == 'depth_noise':
            obs = self.get_obs_depth_noise()
        elif self.perception_type == 'depth':
            obs = self.get_obs_image()
        return obs

    def get_obs_image(self):
        """
        Get depth image and embed state information as a 2-channel observation.
        Channel 0: Processed depth image
        Channel 1: State feature array embedded at top-left corner
        """
        image = self.get_depth_image("FrontDepthCamera")
        image_resize = cv2.resize(image, (self.screen_width, self.screen_height))
        self.min_distance_to_obstacles = image.min()
        image_scaled = np.clip(image_resize, 0, self.max_depth_meters) / self.max_depth_meters * 255
        image_scaled = 255 - image_scaled
        image_uint8 = image_scaled.astype(np.uint8)

        state_feature_array = np.zeros((self.screen_height, self.screen_width))
        state_feature = self.dynamic_model._get_state_feature()
        state_feature_array[0, 0:self.state_feature_length] = state_feature

        image_with_state = np.array([image_uint8, state_feature_array])
        image_with_state = image_with_state.swapaxes(0, 2).swapaxes(0, 1)

        self.feature_all = image_with_state
        return image_with_state

    def get_obs_depth_noise(self):
        """
        Add salt-and-pepper, Gaussian, and motion blur noise to depth image, then embed noisy and clean state features.
        Channel 0: Clean depth
        Channel 1: Noisy depth
        Channel 2: Concatenated clean and noisy state features
        """
        image = self.get_depth_image("FrontDepthCamera")
        image_resize = cv2.resize(image, (self.screen_width, self.screen_height))
        if self.generate_obstacles:
            self.min_distance_to_obstacles = self.get_min_distance_to_obstacles()
        else:
            self.min_distance_to_obstacles = image.min()
        image_scaled = np.clip(image_resize, 0, self.max_depth_meters) / self.max_depth_meters * 255
        image_scaled = 255 - image_scaled
        image_uint8 = image_scaled.astype(np.uint8)

        # Salt-and-pepper noise
        salt_pepper_image = image_uint8.copy()
        num_salt = np.ceil(self.salt_pepper_prob * image_uint8.size * 0.5)
        num_pepper = np.ceil(self.salt_pepper_prob * image_uint8.size * 0.5)
        coords = [np.random.randint(0, i - 1, int(num_salt)) for i in image_uint8.shape]
        salt_pepper_image[coords[0], coords[1]] = 255
        coords = [np.random.randint(0, i - 1, int(num_pepper)) for i in image_uint8.shape]
        salt_pepper_image[coords[0], coords[1]] = 0

        # Gaussian noise
        gaussian_noise = np.random.normal(0, self.gaussian_std, salt_pepper_image.shape)
        gaussian_noise = np.clip(gaussian_noise, -30, 30).astype(np.uint8)
        noisy_image = cv2.add(salt_pepper_image, gaussian_noise)
        noisy_image = np.clip(noisy_image, 0, 255)

        # Motion blur
        kernel_motion_blur = np.zeros((self.motion_blur_kernel_size, self.motion_blur_kernel_size))
        kernel_motion_blur[int((self.motion_blur_kernel_size - 1) / 2), :] = np.ones(self.motion_blur_kernel_size)
        kernel_motion_blur /= self.motion_blur_kernel_size
        blurred_image = cv2.filter2D(noisy_image, -1, kernel_motion_blur)

        # State features and noise
        state_feature_array = np.zeros((self.screen_height, self.screen_width))
        state_feature = self.dynamic_model._get_state_feature()
        state_feature_noise = np.random.normal(0, self.state_feature_noise_std, state_feature.shape)
        state_feature_noise = np.clip(state_feature_noise, -5, 5)
        state_feature_with_noise = np.clip(state_feature + state_feature_noise, 0, 255)

        state_feature_array[0, 0:self.state_feature_length] = state_feature
        state_feature_array[0, self.state_feature_length:2 * self.state_feature_length] = state_feature_with_noise

        image_with_state = np.array([image_uint8, blurred_image, state_feature_array])
        image_with_state = image_with_state.swapaxes(0, 2).swapaxes(0, 1)

        self.feature_all = image_with_state
        return image_with_state

    # ! ---------------------get image-------------------------------------
    def get_depth_image(self, camera_name):
        """
        Continuously request a depth image from the specified camera until a valid image is received.
        Returns:
            depth_meter (ndarray): Depth image in meters.
        """
        while True:
            try:
                responses = self.client.simGetImages([
                    airsim.ImageRequest(camera_name, airsim.ImageType.DepthVis, True)
                ])

                if responses[0].width != 0:
                    depth_img = airsim.list_to_2d_float_array(
                        responses[0].image_data_float, responses[0].width, responses[0].height)
                    depth_meter = depth_img * 100
                    return depth_meter
                else:
                    print("get_image_fail...")

            except msgpackrpc.error.RPCError as e:
                print(f"RPC Error: {e}. Retrying...")
            except Exception as e:
                print(f"Unexpected error: {e}. Retrying...")

            time.sleep(0.1)

    def get_grayscale_image(self, camera_name):
        """
        Continuously request a grayscale image (scene image) until a valid image is received.
        Returns:
            gray_img (ndarray): Grayscale image as 2D float array.
        """
        while True:
            try:
                responses = self.client.simGetImages([
                    airsim.ImageRequest(camera_name, airsim.ImageType.Scene, False)
                ])

                if responses[0].width != 0:
                    gray_img = airsim.list_to_2d_float_array(
                        responses[0].image_data_uint8, responses[0].width, responses[0].height)
                    return gray_img
                else:
                    print("get_image_fail...")

            except msgpackrpc.error.RPCError as e:
                print(f"RPC Error: {e}. Retrying...")
            except Exception as e:
                print(f"Unexpected error: {e}. Retrying...")

            time.sleep(0.1)

    def get_segmentation_image(self, camera_name):
        """
        Continuously request a segmentation image until a valid image is received.
        Converts it to a grayscale segmentation mask.
        Returns:
            segmentation_mask (ndarray): Grayscale segmentation mask.
        """
        while True:
            try:
                responses = self.client.simGetImages([
                    airsim.ImageRequest(camera_name, airsim.ImageType.Segmentation, False, False)
                ])

                if responses[0].width != 0:
                    segmentation_img = np.frombuffer(responses[0].image_data_uint8, dtype=np.uint8)
                    segmentation_img = segmentation_img.reshape(responses[0].height, responses[0].width, 3)
                    segmentation_mask = cv2.cvtColor(segmentation_img, cv2.COLOR_BGR2GRAY)
                    return segmentation_mask
                else:
                    print("get_image_fail...")

            except msgpackrpc.error.RPCError as e:
                print(f"RPC Error: {e}. Retrying...")
            except Exception as e:
                print(f"Unexpected error: {e}. Retrying...")

            time.sleep(0.1)

    def get_depth_gray_image(self):
        """
        Retrieve both depth image and grayscale image from camera "0".
        Converts raw images into depth in meters and resized grayscale.
        Returns:
            depth_meter (ndarray): Depth image in meters.
            img_gray (ndarray): Grayscale image resized to screen dimensions.
        """
        responses = self.client.simGetImages([
            airsim.ImageRequest("0", airsim.ImageType.DepthVis, True),
            airsim.ImageRequest("0", airsim.ImageType.Scene, False, False),
        ])

        while responses[0].width == 0:
            print("get_image_fail...")
            responses = self.client.simGetImages([
                airsim.ImageRequest("0", airsim.ImageType.DepthVis, True),
                airsim.ImageRequest("0", airsim.ImageType.Scene, False, False),
            ])

        depth_img = airsim.list_to_2d_float_array(
            responses[0].image_data_float,
            responses[0].width, responses[0].height)
        depth_meter = depth_img * 100

        img_1d = np.frombuffer(responses[1].image_data_uint8, dtype=np.uint8)
        img_rgb = img_1d.reshape(responses[1].height, responses[1].width, 3)
        img_gray = cv2.cvtColor(img_rgb, cv2.COLOR_BGR2GRAY)
        img_gray = cv2.resize(img_gray, (self.screen_width, self.screen_height))

        return depth_meter, img_gray

    def get_rgb_image(self):
        """
        Continuously request RGB image from camera "0" until a valid one is received.
        Returns:
            rgb_img (ndarray): RGB image reshaped as H x W x 3.
        """
        responses = self.client.simGetImages([
            airsim.ImageRequest("0", airsim.ImageType.Scene, False, False)
        ])

        while responses[0].width == 0:
            print("get_image_fail...")
            responses = self.client.simGetImages([
                airsim.ImageRequest("0", airsim.ImageType.Scene, False, False)
            ])

        rgb_img = np.frombuffer(responses[0].image_data_uint8, dtype=np.uint8)
        rgb_img = rgb_img.reshape(responses[0].height, responses[0].width, 3)
        return rgb_img

    # ! ---------------------calculate rewards-------------------------------------
    def compute_reward_hover(self, done, action_stack):
        """
        Compute reward for hover mode.
        Includes proximity to goal, pose penalties, control effort, and orientation error.
        """
        reward = 0
        reward_success = 2
        reward_crash = -5
        reward_outside = -5
        reward_list = []
        if not done:
            if self.success_flag:
                reward = reward_success
                reward_list = [0, 0, 0]
            elif self.is_reached_goal():
                reward = 2
                reward_list = [0, 0, 0]
            else:
                # 1. Reward for reducing distance to goal
                distance_now = self.get_distance_to_goal_3d()
                reward_distance = self.previous_distance_from_des_point - distance_now
                reward_distance = np.clip(reward_distance, -1, 1)
                self.previous_distance_from_des_point = distance_now

                reward_proximity = (0.5 - distance_now) * 2 if distance_now < 0.5 else 0

                # 2. Penalty for distance deviation in x, y, z
                current_pose = self.dynamic_model.get_position()
                goal_pose = self.dynamic_model.goal_position
                x, y, z = current_pose
                x_g, y_g, z_g = goal_pose
                punishment_x = abs(np.clip((x - x_g) / 2, -1, 1))
                punishment_y = abs(np.clip((y - y_g) / 2, -1, 1))
                punishment_z = abs(np.clip((z - z_g) / 2, -1, 1))
                punishment_pose = punishment_x + punishment_y + punishment_z

                # 3. Penalty for control effort
                vx, vy, vz, v_yaw = action_stack[0]
                normalized_vx = np.clip(vx / self.dynamic_model.v_x_max, -1.0, 1.0)
                normalized_vy = np.clip(vy / self.dynamic_model.v_y_max, -1.0, 1.0)
                normalized_vz = np.clip(vz / self.dynamic_model.v_z_max, -1.0, 1.0)
                normalized_v_yaw = np.clip(v_yaw / self.dynamic_model.yaw_rate_max_rad, -1.0, 1.0)
                punishment_action = normalized_vx ** 2 + normalized_vy ** 2 + normalized_vz ** 2 + normalized_v_yaw ** 2

                # 4. Penalty for orientation error
                curr_roll, curr_pitch, curr_yaw = self.dynamic_model.get_attitude()
                current_quaternion = Rotation.from_euler('xyz', [curr_roll, curr_pitch, curr_yaw]).as_quat()
                goal_quaternion = self.dynamic_model.goal_orientation
                quaternion_error = Rotation.from_quat(goal_quaternion).inv() * Rotation.from_quat(current_quaternion)
                relative_error = quaternion_error.magnitude() / 2
                punishment_orientation = np.clip(relative_error, 0, 1)

                # Combined reward
                reward = 10.0 * reward_distance
                reward_list = [3.0 * reward_distance, -0.3 * punishment_orientation, reward_proximity]
                reward = np.clip(reward, -1, 1)

        else:
            reward_list = [0, 0, 0]
            reward = reward_success if self.success_flag else reward_outside

        return reward, reward_list

    def compute_reward_distance(self, done, action_stack):
        """
        Reward function focused on minimizing 3D distance and direction error to goal,
        while penalizing poor pose, orientation deviation, action jitter, and proximity to obstacles.
        """
        reward = 0
        reward_reach = 10
        reward_crash = -5
        reward_outside = -5
        action = action_stack[0]
        if not done:
            # 1. Reward for reducing distance to goal
            distance_now = self.get_distance_to_goal_3d()
            reward_distance = self.previous_distance_from_des_point - distance_now
            reward_distance = np.clip(reward_distance, -1, 1)
            self.previous_distance_from_des_point = distance_now

            # 2. Reward for reducing yaw angle to goal
            direction_now = self.dynamic_model._get_relative_yaw()
            reward_direction = 10 * (self.previous_direction - direction_now)
            reward_direction = np.clip(reward_direction, -1, 1)
            self.previous_direction = direction_now

            # 3. Pose deviation penalty (x, y, z)
            current_pose = self.dynamic_model.get_position()
            goal_pose = self.dynamic_model.goal_position
            x, y, z = current_pose
            x_g, y_g, z_g = goal_pose
            punishment_xy = np.clip(self.getDis(x, y, 0, 0, x_g, y_g) / 10, 0, 1)
            punishment_z = 2 * abs(np.clip((z - z_g) / 5, -1, 1))
            punishment_pose = punishment_xy + punishment_z

            # 4. Orientation angular error penalty
            error_now = self.dynamic_model._get_vector_angle()
            punishment_angle = abs(np.clip(error_now / (2 * math.pi), -1, 1))

            # 5. Obstacle proximity penalty
            if self.min_distance_to_obstacles < 4:
                punishment_obs = 1 - np.clip((self.min_distance_to_obstacles - self.crash_distance) / 3, 0, 1)
            else:
                punishment_obs = 0

            # 6. Smoothness penalty for action variation
            delta_1 = 0.1 * np.linalg.norm(action_stack[0] - action_stack[1])
            delta_2 = 0.1 * np.linalg.norm(action_stack[1] - action_stack[2])
            punishment_action = np.clip((1.5 * (delta_1 + delta_2)), 0, 1)

            # Final reward combination
            reward = 5.0 * reward_distance - 0.3 * punishment_pose - punishment_angle - punishment_obs
            reward = np.clip(reward / 3, -1, 1)
            reward_list = [5.0 * reward_distance, -0.3 * punishment_pose, - punishment_obs]
        else:
            reward_list = [0, 0, 0]
            if self.step_num < 50:
                reward = -10
            elif self.is_in_desired_pose():
                reward = reward_reach
            elif self.is_not_inside_fly_range():
                reward = reward_outside
            else:
                reward = reward_crash

        return float(reward), reward_list

    def compute_reward_doge(self, done, action_stack):
        """
        Lightweight reward function considering only distance to goal, orientation error, and obstacle proximity.
        """
        reward = 0
        reward_reach = 10
        reward_crash = -5
        reward_outside = -5
        action = action_stack[0]
        if not done:
            # 1. Reward for reducing distance
            distance_now = self.get_distance_to_goal_3d()
            reward_distance = self.previous_distance_from_des_point - distance_now
            reward_distance = np.clip(reward_distance, -1, 1)
            self.previous_distance_from_des_point = distance_now

            # 2. Obstacle distance penalty
            if self.min_distance_to_obstacles < 4:
                punishment_obs = 1 - np.clip((self.min_distance_to_obstacles - self.crash_distance) / 3, 0, 1)
            else:
                punishment_obs = 0

            # 3. Orientation error penalty
            error_now = self.dynamic_model._get_vector_angle()
            punishment_angle = abs(np.clip(error_now / (2 * math.pi), -1, 1))

            # Final reward combination
            reward = 5.0 * reward_distance - punishment_obs - punishment_angle
            reward = np.clip(reward / 3, -1, 1)
            reward_list = [5.0 * reward_distance, - punishment_obs, - punishment_angle]
        else:
            reward_list = [0, 0, 0]
            if self.step_num < 50:
                reward = -10
            elif self.is_in_desired_pose():
                reward = reward_reach
            elif self.is_not_inside_fly_range():
                reward = reward_outside
            else:
                reward = reward_crash

        return float(reward), reward_list

    # ! ---------------------done check-------------------------------------
    def is_done(self):
        """
        Determine whether the episode should be terminated based on:
        - Environment type ('Hover' or 'Nav')
        - Drone flying out of bounds
        - Collision with obstacles
        - Reaching the goal (for 'Nav')
        - Maximum episode steps reached
        """
        if self.env_name == 'Hover':
            episode_done = False
            is_not_inside_workspace_now = self.is_not_inside_fly_range()
            too_close_to_obstable = self.is_crashed()
            episode_done = is_not_inside_workspace_now or \
                           too_close_to_obstable or \
                           self.step_num >= self.max_episode_steps
        elif self.env_name == 'Nav':
            episode_done = False
            is_not_inside_workspace_now = self.is_not_inside_fly_range()
            has_reached_des_pose = self.is_in_desired_pose()
            too_close_to_obstable = self.is_crashed()

            episode_done = is_not_inside_workspace_now or \
                           too_close_to_obstable or \
                           has_reached_des_pose or \
                           self.step_num >= self.max_episode_steps
        return episode_done

    def is_in_desired_pose(self):
        """
        Check whether the drone has reached the desired goal position
        within the acceptable radius threshold.
        """
        in_desired_pose = False
        if self.get_distance_to_goal_3d() < self.accept_radius:
            in_desired_pose = True
        return in_desired_pose

    def is_not_inside_fly_range(self):
        """
        Check whether the drone is flying outside the predefined workspace boundaries.
        Evaluates horizontal (x, y) and vertical (z) limits.
        """
        is_not_inside = False
        current_position = self.dynamic_model.get_position()

        # Compute horizontal Euclidean distance from start position
        distance = math.sqrt(
            (current_position[0] - self.dynamic_model.start_position[0]) ** 2 +
            (current_position[1] - self.dynamic_model.start_position[1]) ** 2
        )

        # If horizontal distance or altitude exceeds workspace range, flag as outside
        if distance > self.fly_range or current_position[2] < self.work_space_z[0] or \
            current_position[2] > self.work_space_z[1]:
            is_not_inside = True

        return is_not_inside

    def is_crashed(self):
        """
        Check whether the drone has collided with any object.
        Uses AirSim collision detection.
        """
        is_crashed = False
        collision_info = self.client.simGetCollisionInfo("SimpleFlight")
        if collision_info.has_collided:
            is_crashed = True

        return is_crashed

    def is_reached_goal(self):
        """
        Check if the drone is within the specified radius of the goal position.
        This is primarily used in the 'Hover' environment.
        """
        current_position = self.dynamic_model.get_position()

        # Compute Euclidean distance to goal position
        distance = math.sqrt(
            (current_position[0] - self.dynamic_model.goal_position[0]) ** 2 +
            (current_position[1] - self.dynamic_model.goal_position[1]) ** 2 +
            (current_position[2] - self.dynamic_model.goal_position[2]) ** 2
        )

        # Check if the drone is close enough to be considered having reached the goal
        if self.env_name == 'Hover':
            if distance < self.hover_radius:
                return True
            else:
                return False

    # ! ----------- useful functions-------------------------------------------
    def get_distance_to_goal_3d(self):
        """
        Calculate 3D Euclidean distance from current position to the goal position.
        """
        current_pose = self.dynamic_model.get_position()
        goal_pose = self.dynamic_model.goal_position
        relative_pose_x = current_pose[0] - goal_pose[0]
        relative_pose_y = current_pose[1] - goal_pose[1]
        relative_pose_z = current_pose[2] - goal_pose[2]

        return math.sqrt(pow(relative_pose_x, 2) + pow(relative_pose_y, 2) + pow(relative_pose_z, 2))

    def getDis(self, pointX, pointY, lineX1, lineY1, lineX2, lineY2):
        """
        Calculate the perpendicular distance between a point and a line segment in 2D.
        Used to compute punishment related to spatial deviation.
        """
        a = lineY2 - lineY1
        b = lineX1 - lineX2
        c = lineX2 * lineY1 - lineX1 * lineY2
        dis = (math.fabs(a * pointX + b * pointY + c)) / (math.pow(a * a + b * b, 0.5))

        return dis

    def set_pyqt_signal_multirotor(self, action, reward, reward_list):
        """
        Emit PyQt signals related to the multirotor's status for visualization or logging:
        - action: current action taken
        - state: current state
        - attitude: orientation angles (e.g., roll, pitch, yaw)
        - reward: current reward and cumulative reward
        - component_signal: individual components of reward
        - pose: target position, start position, current position, and trajectory
        """
        step = int(self.total_step)

        # transfer 2D state and action to 3D
        state = self.dynamic_model.state_raw
        action_output = action
        state_output = state

        self.action_signal.emit(step, action_output)
        self.state_signal.emit(step, state_output)

        # other values
        self.attitude_signal.emit(step, np.asarray(self.dynamic_model.get_attitude(
        )), np.asarray(self.dynamic_model.get_attitude_cmd()))
        self.reward_signal.emit(step, reward, self.cumulated_episode_reward)
        self.component_signal.emit(step, reward_list)
        self.pose_signal.emit(np.asarray(self.dynamic_model.goal_position), np.asarray(
            self.dynamic_model.start_position), np.asarray(self.dynamic_model.get_position()),
                              np.asarray(self.trajectory_list))
