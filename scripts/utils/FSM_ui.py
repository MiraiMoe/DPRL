import sys
import math
import numpy as np
import pyqtgraph as pg
from PyQt5 import QtWidgets
from PIL import Image
from PyQt5.QtWidgets import QGroupBox, QHBoxLayout, QVBoxLayout, QWidget
from pyqtgraph.widgets.MatplotlibWidget import MatplotlibWidget
from configparser import ConfigParser
import os


class TrainingUi(QWidget):
    """
    PyQt5 GUI used to visualize data during UAV training.
    Includes plots for actions, states, attitude, rewards, and trajectory.
    """

    def __init__(self, config):
        super(TrainingUi, self).__init__()
        self.cfg = ConfigParser()
        self.cfg.read(config)

        self.init_ui()
        self.select_flag = True
        self.curves = []

    def init_ui(self):
        """
        Initialize the training UI layout and plotting widgets.
        Includes four main components:
            - Action plot
            - State feature plot
            - Attitude plot
            - Reward & Trajectory plot
        """
        self.setWindowTitle("Training UI")

        pg.setConfigOptions(leftButtonPan=False)
        pg.setConfigOption('background', 'w')
        pg.setConfigOption('foreground', 'k')
        pg.setConfigOption('imageAxisOrder', 'row-major')

        self.max_len = 100
        self.ptr = -self.max_len

        self.dynamics = self.cfg.get('options', 'dynamic_name')

        action_plot_gb = self.create_actionPlot_groupBox_multirotor()
        state_plot_gb = self.create_state_plot_groupbox()
        attitude_plot_gb = self.create_attitude_plot_groupbox()
        reward_plot_gb = self.create_reward_plot_groupbox()
        component_plot_gb = self.create_component_plot_groupbox()
        traj_plot_gb = self.create_traj_plot_groupbox()

        right_widget = QWidget()
        vlayout = QVBoxLayout()
        vlayout.addWidget(reward_plot_gb)
        vlayout.addWidget(component_plot_gb)
        vlayout.addWidget(traj_plot_gb)
        right_widget.setLayout(vlayout)

        main_layout = QHBoxLayout()
        main_layout.addWidget(action_plot_gb)
        main_layout.addWidget(state_plot_gb)
        main_layout.addWidget(attitude_plot_gb)
        main_layout.addWidget(right_widget)

        self.setLayout(main_layout)

        self.pen_red = pg.mkPen(color='r', width=2)     # For real data
        self.pen_blue = pg.mkPen(color='b', width=1)    # For command data
        self.pen_green = pg.mkPen(color='g', width=2)   # For selected trajectory

    def update_value_list(self, value_list, value):
        """
        Update the data list by shifting left and adding a new value.
        """
        value_list[:-1] = value_list[1:]
        value_list[-1] = float(value)
        return value_list

    def create_actionPlot_groupBox_multirotor(self):
        """
        Create a groupbox for visualizing multirotor actions including:
            - vx, vy, vz (m/s)
            - yaw_rate (deg/s)
        """
        actionPlotGroupBox = QGroupBox('Action (multirotor)')

        # Initialize buffers
        self.v_x_cmd_list = np.zeros(self.max_len)
        self.v_x_real_list = np.zeros(self.max_len)
        self.v_y_cmd_list = np.zeros(self.max_len)
        self.v_y_real_list = np.zeros(self.max_len)
        self.v_z_cmd_list = np.zeros(self.max_len)
        self.v_z_real_list = np.zeros(self.max_len)
        self.yaw_rate_cmd_list = np.zeros(self.max_len)
        self.yaw_rate_real_list = np.zeros(self.max_len)

        layout = QVBoxLayout()

        # vx plot
        self.plotWidget_v_x = pg.PlotWidget(title='v_x (m/s)')
        self.plotWidget_v_x.setYRange(
            max=self.cfg.getfloat('multirotor_new', 'v_x_max'),
            min=-self.cfg.getfloat('multirotor_new', 'v_x_max'))
        self.plotWidget_v_x.showGrid(x=True, y=True)
        self.plot_v_x = self.plotWidget_v_x.plot()
        self.plot_v_x_cmd = self.plotWidget_v_x.plot()
        layout.addWidget(self.plotWidget_v_x)

        # vy plot
        self.plotWidget_v_y = pg.PlotWidget(title='v_y (m/s)')
        self.plotWidget_v_y.setYRange(
            max=self.cfg.getfloat('multirotor_new', 'v_y_max'),
            min=-self.cfg.getfloat('multirotor_new', 'v_y_max'))
        self.plotWidget_v_y.showGrid(x=True, y=True)
        self.plot_v_y = self.plotWidget_v_y.plot()
        self.plot_v_y_cmd = self.plotWidget_v_y.plot()
        layout.addWidget(self.plotWidget_v_y)

        # vz plot
        self.plotWidget_v_z = pg.PlotWidget(title='v_z (m/s)')
        self.plotWidget_v_z.setYRange(
            max=self.cfg.getfloat('multirotor_new', 'v_z_max'),
            min=-self.cfg.getfloat('multirotor_new', 'v_z_max'))
        self.plotWidget_v_z.showGrid(x=True, y=True)
        self.plot_v_z = self.plotWidget_v_z.plot()
        self.plot_v_z_cmd = self.plotWidget_v_z.plot()
        layout.addWidget(self.plotWidget_v_z)

        # yaw_rate plot
        self.plotWidget_yaw_rate = pg.PlotWidget(title='yaw_rate (deg/s)')
        self.plotWidget_yaw_rate.setYRange(
            max=self.cfg.getfloat('multirotor_new', 'yaw_rate_max_deg'),
            min=-self.cfg.getfloat('multirotor_new', 'yaw_rate_max_deg'))
        self.plotWidget_yaw_rate.showGrid(x=True, y=True)
        self.plot_yaw_rate = self.plotWidget_yaw_rate.plot()
        self.plot_yaw_rate_cmd = self.plotWidget_yaw_rate.plot()
        layout.addWidget(self.plotWidget_yaw_rate)

        actionPlotGroupBox.setLayout(layout)
        return actionPlotGroupBox

    def action_cb(self, step, action):
        """
        Callback function to update action plots.
        """
        self.action_cb_multirotor(step, action)

    def action_cb_multirotor(self, step, action):
        """
        Update command action plots for vx, vy, vz, yaw_rate.
        """
        self.update_value_list(self.v_x_cmd_list, action[0])
        self.update_value_list(self.v_y_cmd_list, action[1])
        self.update_value_list(self.v_z_cmd_list, -action[2])
        self.update_value_list(self.yaw_rate_cmd_list, math.degrees(action[3]))

        self.plot_v_x_cmd.setData(self.v_x_cmd_list, pen=self.pen_blue)
        self.plot_v_y_cmd.setData(self.v_y_cmd_list, pen=self.pen_blue)
        self.plot_v_z_cmd.setData(self.v_z_cmd_list, pen=self.pen_blue)
        self.plot_yaw_rate_cmd.setData(self.yaw_rate_cmd_list, pen=self.pen_blue)

    def create_state_plot_groupbox(self):
        """
        Create a groupbox for visualizing state features:
            - distance_x, distance_y, distance_z (m)
        """
        state_plot_groupbox = QGroupBox(title='State feature')

        self.distance_x_list = np.zeros(self.max_len)
        self.distance_y_list = np.zeros(self.max_len)
        self.distance_z_list = np.zeros(self.max_len)

        layout = QVBoxLayout()

        self.pw_distance_x = pg.PlotWidget(title='distance_x (m)')
        self.pw_distance_x.showGrid(x=True, y=True)
        self.plot_distance_x = self.pw_distance_x.plot()

        self.pw_distance_y = pg.PlotWidget(title='distance_y (m)')
        self.pw_distance_y.showGrid(x=True, y=True)
        self.plot_distance_y = self.pw_distance_y.plot()

        self.pw_distance_z = pg.PlotWidget(title='distance_z (m)')
        self.pw_distance_z.showGrid(x=True, y=True)
        self.plot_distance_z = self.pw_distance_z.plot()

        layout.addWidget(self.pw_distance_x)
        layout.addWidget(self.pw_distance_y)
        layout.addWidget(self.pw_distance_z)

        state_plot_groupbox.setLayout(layout)
        return state_plot_groupbox

    def state_cb(self, step, state_raw):
        """
        Callback function to update state plots:
            - Distance to target (x, y, z)
            - Real velocities and yaw rate
        """
        self.update_value_list(self.distance_x_list, abs(state_raw[3]))
        self.update_value_list(self.distance_y_list, abs(state_raw[4]))
        self.update_value_list(self.distance_z_list, state_raw[5])

        self.plot_distance_x.setData(self.distance_x_list, pen=self.pen_red)
        self.plot_distance_y.setData(self.distance_y_list, pen=self.pen_red)
        self.plot_distance_z.setData(self.distance_z_list, pen=self.pen_red)

        # Update real action feedback
        self.update_value_list(self.v_x_real_list, state_raw[6])
        self.update_value_list(self.v_y_real_list, state_raw[7])
        self.update_value_list(self.v_z_real_list, state_raw[8])
        self.update_value_list(self.yaw_rate_real_list, state_raw[11])

        self.plot_v_x.setData(self.v_x_real_list, pen=self.pen_red)
        self.plot_v_y.setData(self.v_y_real_list, pen=self.pen_red)
        self.plot_v_z.setData(self.v_z_real_list, pen=self.pen_red)
        self.plot_yaw_rate.setData(self.yaw_rate_real_list, pen=self.pen_red)

    def create_attitude_plot_groupbox(self):
        """
        Create a groupbox for plotting UAV attitude:
            - roll, pitch, yaw (in degrees)
            - comparison between actual and commanded values
        """
        plot_gb = QGroupBox(title='Attitude')
        layout = QVBoxLayout()

        self.roll_list = np.zeros(self.max_len)
        self.roll_cmd_list = np.zeros(self.max_len)
        self.pitch_list = np.zeros(self.max_len)
        self.pitch_cmd_list = np.zeros(self.max_len)
        self.yaw_list = np.zeros(self.max_len)
        self.yaw_cmd_list = np.zeros(self.max_len)

        # Roll
        self.pw_roll = pg.PlotWidget(title='roll (deg)')
        self.pw_roll.setYRange(max=45, min=-45)
        self.pw_roll.showGrid(x=True, y=True)
        self.plot_roll = self.pw_roll.plot()
        self.plot_roll_cmd = self.pw_roll.plot()

        # Pitch
        self.pw_pitch = pg.PlotWidget(title='pitch (deg)')
        self.pw_pitch.setYRange(max=25, min=-25)
        self.pw_pitch.showGrid(x=True, y=True)
        self.plot_pitch = self.pw_pitch.plot()
        self.plot_pitch_cmd = self.pw_pitch.plot()

        # Yaw
        self.pw_yaw = pg.PlotWidget(title='yaw (deg)')
        self.pw_yaw.showGrid(x=True, y=True)
        self.plot_yaw = self.pw_yaw.plot()
        self.plot_yaw_cmd = self.pw_yaw.plot()

        layout.addWidget(self.pw_roll)
        layout.addWidget(self.pw_pitch)
        layout.addWidget(self.pw_yaw)

        plot_gb.setLayout(layout)
        return plot_gb

    def attitude_plot_cb(self, step, attitude, attitude_cmd):
        """
        Callback to update attitude plots with actual values.
        """
        self.update_value_list(self.pitch_list, math.degrees(attitude[0]))
        self.update_value_list(self.roll_list, math.degrees(attitude[1]))
        self.update_value_list(self.yaw_list, math.degrees(attitude[2]))

        self.plot_pitch.setData(self.pitch_list, pen=self.pen_red)
        self.plot_roll.setData(self.roll_list, pen=self.pen_red)
        self.plot_yaw.setData(self.yaw_list, pen=self.pen_red)

    def create_reward_plot_groupbox(self):
        """
        Create a groupbox for plotting rewards:
            - Immediate reward
            - Accumulated reward over time
        """
        reward_plot_groupbox = QGroupBox(title='Reward')
        layout = QHBoxLayout()
        reward_plot_groupbox.setFixedWidth(600)

        self.reward_list = np.zeros(self.max_len)
        self.total_reward_list = np.zeros(self.max_len)

        self.rw_pw_1 = pg.PlotWidget(title='reward')
        self.rw_pw_1.showGrid(x=True, y=True)
        self.rw_p_1 = self.rw_pw_1.plot()

        self.rw_pw_2 = pg.PlotWidget(title='total reward')
        self.rw_pw_2.showGrid(x=True, y=True)
        self.rw_p_2 = self.rw_pw_2.plot()

        layout.addWidget(self.rw_pw_1)
        layout.addWidget(self.rw_pw_2)

        reward_plot_groupbox.setLayout(layout)
        return reward_plot_groupbox

    def reward_plot_cb(self, step, reward, total_reward):
        """
        Callback to update reward plots with current step values.
        """
        self.update_value_list(self.reward_list, reward)
        self.update_value_list(self.total_reward_list, total_reward)

        self.rw_p_1.setData(self.reward_list, pen=self.pen_red)
        self.rw_p_2.setData(self.total_reward_list, pen=self.pen_red)

    def create_component_plot_groupbox(self):
        """
        Create a groupbox for visualizing individual reward components:
            - reward_distance: reward based on distance to the goal
            - punishment_pose: penalty for poor pose/orientation
            - punishment_action: penalty for aggressive or unstable actions
        """
        component_plot_groupbox = QGroupBox(title='Reward Components')
        layout = QHBoxLayout()
        component_plot_groupbox.setFixedWidth(600)

        # Initialize time-series data buffers
        self.reward_1_list = np.zeros(self.max_len)
        self.reward_2_list = np.zeros(self.max_len)
        self.reward_3_list = np.zeros(self.max_len)

        # Create plots for each reward component
        self.rw_pw_3 = pg.PlotWidget(title='reward_distance')
        self.rw_pw_3.showGrid(x=True, y=True)
        self.rw_p_3 = self.rw_pw_3.plot()

        self.rw_pw_4 = pg.PlotWidget(title='punishment_pose')
        self.rw_pw_4.showGrid(x=True, y=True)
        self.rw_p_4 = self.rw_pw_4.plot()

        self.rw_pw_5 = pg.PlotWidget(title='punishment_action')
        self.rw_pw_5.showGrid(x=True, y=True)
        self.rw_p_5 = self.rw_pw_5.plot()

        # Add plots to layout
        layout.addWidget(self.rw_pw_3)
        layout.addWidget(self.rw_pw_4)
        layout.addWidget(self.rw_pw_5)

        component_plot_groupbox.setLayout(layout)
        return component_plot_groupbox

    def component_plot_cb(self, step, reward_component):
        """
        Callback to update reward component plots with latest values.

        Args:
            step (int): Current step count.
            reward_component (list or np.ndarray): Three components of reward:
                [reward_distance, punishment_pose, punishment_action]
        """
        self.update_value_list(self.reward_1_list, reward_component[0])
        self.update_value_list(self.reward_2_list, reward_component[1])
        self.update_value_list(self.reward_3_list, reward_component[2])

        self.rw_p_3.setData(self.reward_1_list, pen=self.pen_red)
        self.rw_p_4.setData(self.reward_2_list, pen=self.pen_red)
        self.rw_p_5.setData(self.reward_3_list, pen=self.pen_red)

    def create_traj_plot_groupbox(self):
        """
        Create a groupbox for visualizing 2D trajectory plots of UAV:
            - Includes background map
            - Start point, goal point, and full trajectory line
        """
        traj_plot_groupbox = QGroupBox('Trajectory')
        traj_plot_groupbox.setFixedSize(600, 600)
        layout = QVBoxLayout()

        self.traj_pw = pg.PlotWidget(title='trajectory')
        self.traj_pw.showGrid(x=True, y=True)
        self.traj_pw.setXRange(max=140, min=-140)
        self.traj_pw.setYRange(max=140, min=-140)
        self.traj_pw.invertY()  # Invert Y-axis to match image coordinates

        if self.cfg.get('options', 'env_name') == 'Hover':
            self.traj_pw.setXRange(max=60, min=-60)
            self.traj_pw.setYRange(max=60, min=-60)

        self.traj_plot = self.traj_pw.plot()
        layout.addWidget(self.traj_pw)
        traj_plot_groupbox.setLayout(layout)

        return traj_plot_groupbox

    def traj_plot_cb(self, goal, start, current_pose, trajectory_list):
        """
        Callback to plot UAV trajectory in XY-plane along with start and goal positions.

        Args:
            goal (list): [x, y] coordinates of the goal.
            start (list): [x, y] coordinates of the start position.
            current_pose (list): Current pose, not used directly here.
            trajectory_list (np.ndarray): N x 2 array of [x, y] trajectory points.
        """

        # Add background image once (for applicable environments)
        background_list = ['SimpleAvoid', 'NH_center', 'City_400', 'Tree_200', 'Forest', 'SimpleForest']
        if self.cfg.get('options', 'env_name') in background_list:
            if hasattr(self, 'background_img') and self.background_img not in self.traj_pw.items():
                self.traj_pw.addItem(self.background_img)

        # Plot start point
        if not hasattr(self, 'start_plot'):
            self.start_plot = self.traj_pw.plot([start[0]], [start[1]], symbol='o', pen=None, symbolBrush='b')
        else:
            self.start_plot.setData([start[0]], [start[1]])

        # Plot goal point
        if not hasattr(self, 'goal_plot'):
            self.goal_plot = self.traj_pw.plot([goal[0]], [goal[1]], symbol='o', pen=None, symbolBrush='r')
        else:
            self.goal_plot.setData([goal[0]], [goal[1]])

        # Plot current trajectory
        pen = self.pen_red if self.select_flag else self.pen_green
        if not hasattr(self, 'trajectory_plot'):
            self.trajectory_plot = self.traj_pw.plot(trajectory_list[..., 0], trajectory_list[..., 1], pen=pen)
        else:
            self.trajectory_plot.setData(trajectory_list[..., 0], trajectory_list[..., 1])
            self.trajectory_plot.setPen(pen)

        # Optional: store the latest trajectory in a list
        if not hasattr(self, 'trajectory_curves'):
            self.trajectory_curves = []
        self.trajectory_curves.append(self.trajectory_plot)

