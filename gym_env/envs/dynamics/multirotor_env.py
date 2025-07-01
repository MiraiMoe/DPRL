import airsim
import numpy as np
import math
from gym import spaces
import random
from scipy.spatial.transform import Rotation as R

np.random.seed(0)

class MultirotorDynamicsEnv():
    '''
    A simplified multirotor dynamics interface for vision-based navigation (applying MAE strategy),
    using AirSim's SimpleFlight mode.

    Control command is sent via:
    client.moveByVelocityZAsync(v_x, v_y, v_z, yaw_rate)
    '''

    def __init__(self, cfg, vehicle_name="SimpleFlight") -> None:
        # === Basic Parameters ===
        self.dt = cfg.getfloat('multirotor_new', 'dt')
        self.vehicle_name = vehicle_name

        # === Environment Setup ===
        self.env_name = cfg.get('options', 'env_name')
        self.privileged_info = cfg.get('options', 'privileged_info')

        # === Start and Goal Position/Orientation ===
        self.start_position = [0, 0, 0]
        self.start_random_angle = None
        self.goal_position = [0, 0, 0]
        self.goal_orientation = R.from_euler('xyz', [0, 0, 0]).as_quat()
        self.goal_distance = None
        self.goal_random_angle = None

        # === Drone State Variables ===
        self.x = self.y = self.z = 0
        self.v_x = self.v_y = self.v_z = 0
        self.dx = self.dy = self.dz = 0
        self.yaw = self.dyaw = self.yaw_rate = 0
        self.v_xy = 0

        # === State Feature Dimensions ===
        if self.env_name == 'Hover':
            self.state_feature_length = 14
        elif self.env_name == 'Nav':
            self.state_feature_length = 8
        else:
            raise Exception("Invalid env_name!", self.env_name)

        # === Action Constraints ===
        self.acc_x_max = cfg.getfloat('multirotor_new', 'acc_x_max')
        self.acc_y_max = cfg.getfloat('multirotor_new', 'acc_y_max')
        self.v_x_max = cfg.getfloat('multirotor_new', 'v_x_max')
        self.v_y_max = cfg.getfloat('multirotor_new', 'v_y_max')
        self.v_x_min = cfg.getfloat('multirotor_new', 'v_x_min')
        self.v_y_min = cfg.getfloat('multirotor_new', 'v_y_min')
        self.v_z_max = cfg.getfloat('multirotor_new', 'v_z_max')
        self.yaw_rate_max_deg = cfg.getfloat('multirotor_new', 'yaw_rate_max_deg')
        self.yaw_rate_max_rad = cfg.getfloat('multirotor_new', 'yaw_rate_max_rad')
        self.v_xy_max = self.v_x_max
        self.v_xy_min = self.v_x_min

        # === Environment Boundaries ===
        self.range_x = cfg.getfloat('multirotor_new', 'range_x')
        self.range_y = cfg.getfloat('multirotor_new', 'range_y')
        self.range_z = cfg.getfloat('multirotor_new', 'range_z')
        self.diff_x_max = cfg.getfloat('multirotor_new', 'diff_x_max')
        self.diff_y_max = cfg.getfloat('multirotor_new', 'diff_y_max')
        self.diff_z_max = cfg.getfloat('multirotor_new', 'diff_z_max')
        self.max_vertical_difference = self.diff_z_max

        # === Logging and Evaluation ===
        self.previous_action = np.zeros(4)
        self.current_action = np.zeros(4)
        self.last_restart_steps = 0
        self.time_eval = cfg.getint('multirotor_new', 'time_eval')
        self.timesteps = 0
        self.state_log = []
        self.angle_list = np.linspace(0, 2 * np.pi, 30, endpoint=False).tolist()

        # === Action Space Definition ===
        self.action_space = spaces.Box(
            low=np.array([
                -self.v_x_max, -self.v_y_max, -self.v_z_max, -self.yaw_rate_max_rad]),
            high=np.array([
                self.v_x_max, self.v_y_max, self.v_z_max, self.yaw_rate_max_rad]),
            dtype=np.float32
        )

    def connect(self, port):
        """
        Establish connection with the AirSim simulator using the specified port.

        This function initializes the AirSim MultirotorClient, confirms the connection,
        and enables API control for the specified vehicle. It also arms the drone,
        preparing it for takeoff or movement commands.

        Args:
            port (int): The port number used to connect to the AirSim simulation instance.
        """
        self.port = port
        # Initialize AirSim multirotor client with specified port
        self.client = airsim.MultirotorClient(port=self.port)
        # Confirm connection to the AirSim environment
        self.client.confirmConnection()
        # Enable API control for the given vehicle
        self.client.enableApiControl(True, vehicle_name=self.vehicle_name)
        # Arm the vehicle to allow movement commands
        self.client.armDisarm(True, vehicle_name=self.vehicle_name)

    def reset(self):
        """
        Reset drone to the start position with orientation facing the goal direction.
        Reconnect, arm, and ascend to start altitude in AirSim.
        """
        self.client.reset()

        # Calculate yaw angle toward the goal
        yaw_noise = math.atan2(
            self.goal_position[1] - self.start_position[1],
            self.goal_position[0] - self.start_position[0]
        )

        # Set drone pose in simulation
        pose = self.client.simGetVehiclePose(vehicle_name=self.vehicle_name)
        pose.position.x_val = self.start_position[0]
        pose.position.y_val = self.start_position[1]
        pose.position.z_val = -self.start_position[2]  # AirSim uses NED
        pose.orientation = airsim.to_quaternion(0, 0, yaw_noise)
        self.client.simSetVehiclePose(pose, ignore_collision=True, vehicle_name=self.vehicle_name)

        # Unpause simulation, arm and takeoff
        self.client.simPause(False)
        self.client.enableApiControl(True, vehicle_name=self.vehicle_name)
        self.client.armDisarm(True, vehicle_name=self.vehicle_name)
        self.client.moveToZAsync(
            -self.start_position[2], velocity=2.0, vehicle_name=self.vehicle_name
        ).join()
        self.client.simPause(True)

    def set_action(self, action):
        """
        Apply a velocity and yaw rate action to the drone in AirSim.

        Args:
            action (array-like): [v_x, v_y, v_z, yaw_rate]
        """
        self.previous_action = self.current_action

        # Parse and apply new action
        v_x, v_y, v_z, yaw_rate = map(float, action)
        self.current_action = np.array([v_x, v_y, v_z, yaw_rate])

        # Issue movement command to simulator
        self.client.simPause(False)
        self.client.moveByVelocityAsync(
            v_x, v_y, v_z, duration=self.dt,
            drivetrain=airsim.DrivetrainType.MaxDegreeOfFreedom,
            yaw_mode=airsim.YawMode(is_rate=True, yaw_or_rate=math.degrees(yaw_rate)),
            vehicle_name=self.vehicle_name
        ).join()
        self.client.simPause(True)

    def set_start(self, obstacle_range, work_space_x, work_space_y):
        """
        Randomly generate a valid start position that does NOT lie within the given obstacle annulus ranges.

        Args:
            obstacle_range: List of two [min_radius, max_radius] pairs.
                            The generated position must NOT lie within either of the circular obstacle bands.
            work_space_x: Tuple or list specifying (min_x, max_x) workspace limits along x-axis.
            work_space_y: Tuple or list specifying (min_y, max_y) workspace limits along y-axis.
        """
        if self.env_name == 'Hover':
            while True:
                # Randomly sample x and y within workspace
                x = random.uniform(work_space_x[0], work_space_x[1])
                y = random.uniform(work_space_y[0], work_space_y[1])
                distance_to_origin = math.sqrt(x ** 2 + y ** 2)

                # Check if the distance avoids both obstacle annulus regions
                if (obstacle_range[0][1] < distance_to_origin < obstacle_range[1][0]) or \
                        (distance_to_origin > obstacle_range[1][1]):
                    break
            z = random.uniform(3, 8)  # Random altitude
        elif self.env_name == 'Nav':
            x, y = 0, 0  # Fixed start for navigation task
            z = random.uniform(3, 8)

        self.start_position = [x, y, z]

    def generate_random_point_on_circle(self, x_center, y_center, radius=5):
        """
        Generate a random point on a circle with given center and radius.

        Args:
            x_center, y_center: Coordinates of the circle center.
            radius: Radius of the circle (default = 5).

        Returns:
            Tuple of (x, y) representing a point on the circle.
        """
        theta = random.uniform(0, 2 * math.pi)
        x = x_center + radius * math.cos(theta)
        y = y_center + radius * math.sin(theta)
        return x, y

    def get_valid_point_on_circle(self):
        """
        Generate a valid start point on a circle centered at the target, ensuring no obstacles exist along the line of sight.

        Returns:
            (x, y, z): A valid point in 3D space from which the drone has direct visibility to the target.
        """
        radius = 3  # Fixed distance from target
        while True:
            # Get target position from simulation
            target_pose = self.client.simGetObjectPose(self.target_name)
            x_center = target_pose.position.x_val
            y_center = target_pose.position.y_val
            z_center = target_pose.position.z_val

            # Sample a point on the circle
            x, y = self.generate_random_point_on_circle(x_center, y_center, radius)
            z = 5  # Fixed altitude for start point

            # Line-of-sight check
            start_point = airsim.Vector3r(x, y, z)
            end_point = airsim.Vector3r(x_center, y_center, z_center)
            if self.client.simTestLineOfSightBetweenPoints(start_point, end_point):
                return x, y, z

    def set_goal(self, goal_distance):
        """
        Generate a goal position at a specific distance and random angle from the current start position.
        Also calculates the goal orientation (yaw).

        Args:
            goal_distance: Desired Euclidean distance between start and goal.
        """
        self.goal_random_angle = random.uniform(0, 2 * math.pi)
        start_x, start_y, start_z = self.start_position

        goal_x = start_x + goal_distance * math.cos(self.goal_random_angle)
        goal_y = start_y + goal_distance * math.sin(self.goal_random_angle)
        goal_z = start_z  # Keep goal and start at same altitude

        self.goal_position = [goal_x, goal_y, goal_z]

        # Compute yaw from start to goal
        delta_x = goal_x - start_x
        delta_y = goal_y - start_y
        yaw_angle = math.atan2(delta_y, delta_x)

        # Convert yaw to quaternion (roll and pitch = 0)
        self.goal_orientation = self.euler_to_quaternion(0, 0, yaw_angle)

    def euler_to_quaternion(self, roll, pitch, yaw):
        """
        Convert Euler angles to quaternion format.

        Args:
            roll: Rotation about X-axis
            pitch: Rotation about Y-axis
            yaw: Rotation about Z-axis

        Returns:
            Quaternion as np.ndarray
        """
        rotation = R.from_euler('xyz', [roll, pitch, yaw])
        return rotation.as_quat()

    def _get_state_feature_raw(self):
        """
        Compose the raw state feature vector based on current position, velocity, attitude,
        and relative position to goal.

        Returns:
            np.ndarray containing the 14-dimensional state vector.
        """
        current_position = self.get_position()
        x, y, z = current_position

        # Distance to goal
        dx = self.goal_position[0] - x
        dy = self.goal_position[1] - y
        dz = self.goal_position[2] - z

        # Velocity and yaw rate
        velocity = self.get_velocity()
        vx, vy, vz, yaw_rate = velocity

        # Attitude angles (in radians)
        pitch, roll, yaw = self.get_attitude()
        relative_yaw = self._get_relative_yaw()

        # Compose state vector (angles converted to degrees)
        state_raw = np.array([
            x, y, z,
            dx, dy, dz,
            vx, vy, vz,
            math.degrees(pitch),
            math.degrees(roll),
            math.degrees(yaw),
            math.degrees(relative_yaw),
            math.degrees(yaw_rate)
        ])
        return state_raw

    def _get_state_feature(self):
        """
        Update and retrieve the current UAV state in normalized form.

        Returns:
            state_norm (np.ndarray): Normalized state vector in the range [0, 255].
        """
        # Get current position
        current_position = self.get_position()
        x, y, z = current_position

        # Compute distance difference to goal
        dx = self.goal_position[0] - x
        dy = self.goal_position[1] - y
        dz = self.goal_position[2] - z

        # Get velocity and angular velocity (yaw rate)
        velocity_x, velocity_y, velocity_z, yaw_rate = self.get_velocity()

        # Get current pitch, roll, yaw
        pitch, roll, yaw = self.get_attitude()

        # Compute relative yaw between current heading and goal
        relative_yaw = self._get_relative_yaw()

        # Normalize each part of the state vector (scaled to [0, 255])
        distance_diff_norm = np.array([dx / self.diff_x_max, dy / self.diff_y_max, dz / self.diff_z_max])
        distance_diff_norm = (distance_diff_norm + 1) * 255 / 2

        velocity_norm = np.array([velocity_x / self.v_x_max, velocity_y / self.v_y_max, velocity_z / self.v_z_max])
        velocity_norm = (velocity_norm + 1) * 255 / 2

        relative_yaw_norm = (relative_yaw / (math.pi / 2) + 1) * 255 / 2
        yaw_rate_norm = (yaw_rate / self.yaw_rate_max_rad + 1) * 255 / 2

        position_norm = np.array([x / self.range_x, y / self.range_y, z / self.range_z])

        # Extract rotation vector from quaternion (first two columns of rotation matrix)
        rotation = self.get_rotation_vector()
        rotation_norm = (rotation + 1) * 255 / 2

        # Store raw state (mostly for logging/debugging)
        self.state_raw = np.array([
            x, y, z, dx, dy, dz,
            velocity_x, velocity_y, velocity_z,
            math.degrees(pitch), math.degrees(roll), math.degrees(yaw),
            math.degrees(relative_yaw), math.degrees(yaw_rate)
        ])

        # Construct normalized state vector based on environment and privilege level
        if self.env_name == 'Nav':
            state_norm = np.concatenate((distance_diff_norm, velocity_norm, [relative_yaw_norm, yaw_rate_norm]))

        elif self.env_name == 'Hover':
            state_norm = np.concatenate((distance_diff_norm, velocity_norm,
                                             [relative_yaw_norm, yaw_rate_norm], rotation_norm))

        state_norm = np.clip(state_norm, 0, 255)  # Ensure range is within [0, 255]
        self.state_norm = state_norm

        return state_norm

    def _get_vector_angle(self):
        """
        Compute the angular difference between current yaw and velocity direction.

        Returns:
            yaw_error (float): Angular error in radians within [-pi, pi].
        """
        velocity_x, velocity_y, _, _ = self.get_velocity()
        angle = math.atan2(velocity_y, velocity_x)
        yaw_current = self.get_attitude()[2]
        yaw_error = angle - yaw_current

        if yaw_error > math.pi:
            yaw_error -= 2 * math.pi
        elif yaw_error < -math.pi:
            yaw_error += 2 * math.pi

        return yaw_error

    def _get_relative_yaw(self):
        """
        Compute the relative yaw between the current position and the goal.

        Returns:
            yaw_error (float): Angular difference from UAV heading to goal in radians within [-pi, pi].
        """
        current_position = self.get_position()
        dx = self.goal_position[0] - current_position[0]
        dy = self.goal_position[1] - current_position[1]
        angle_to_goal = math.atan2(dy, dx)

        yaw_current = self.get_attitude()[2]
        yaw_error = angle_to_goal - yaw_current

        if yaw_error > math.pi:
            yaw_error -= 2 * math.pi
        elif yaw_error < -math.pi:
            yaw_error += 2 * math.pi

        return yaw_error

    def get_position(self):
        """
        Get the UAV's current position in world coordinates.

        Returns:
            [x, y, z] (list): Position in meters (airsim's z is downward, so we flip the sign).
        """
        pos = self.client.simGetVehiclePose(vehicle_name=self.vehicle_name).position
        return [pos.x_val, pos.y_val, -pos.z_val]

    def get_velocity(self):
        """
        Get the UAV's current velocity and yaw rate.

        Returns:
            [vx, vy, vz, yaw_rate] (list): Linear velocities and angular velocity around Z.
        """
        state = self.client.getMultirotorState(vehicle_name=self.vehicle_name)
        lv = state.kinematics_estimated.linear_velocity
        av = state.kinematics_estimated.angular_velocity
        return [lv.x_val, lv.y_val, -lv.z_val, av.z_val]

    def get_attitude(self):
        """
        Get the UAV's current attitude (Euler angles).

        Returns:
            (pitch, roll, yaw) (tuple): Orientation angles in radians.
        """
        orientation = self.client.simGetVehiclePose(vehicle_name=self.vehicle_name).orientation
        self.state_current_attitude = orientation
        return airsim.to_eularian_angles(orientation)

    def get_distance_to_goal_2d(self):
        """
        Compute the 2D Euclidean distance to the goal.

        Returns:
            distance (float): Distance in XY plane.
        """
        x, y, _ = self.get_position()
        gx, gy, _ = self.goal_position
        return math.sqrt((x - gx) ** 2 + (y - gy) ** 2)

    def quaternion_to_rotation_matrix(self, quaternion):
        """
        Convert a quaternion to a 3x3 rotation matrix.

        Args:
            quaternion (airsim.Quaternionr): Quaternion to convert.

        Returns:
            rotation_matrix (np.ndarray): 3x3 rotation matrix.
        """
        r = R.from_quat([quaternion.x_val, quaternion.y_val, quaternion.z_val, quaternion.w_val])
        return r.as_matrix()

    def get_rotation_vector(self):
        """
        Extract the first two columns of the UAV's rotation matrix as a flattened 1x6 vector.

        Returns:
            rotation_vector (np.ndarray): Flattened rotation vector [r11, r21, r31, r12, r22, r32].
        """
        quaternion = self.client.simGetVehiclePose(vehicle_name=self.vehicle_name).orientation
        self.state_current_quaternion = quaternion
        rotation_matrix = self.quaternion_to_rotation_matrix(quaternion)
        rotation_vector = rotation_matrix[:, :2].flatten()
        return rotation_vector

    def get_attitude_cmd(self):
        """
        Placeholder: Return current yaw command value.

        Returns:
            list: Currently fixed to [0.0, 0.0, self.yaw]
        """
        return [0.0, 0.0, self.yaw]