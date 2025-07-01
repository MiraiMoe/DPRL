import sys
import os
from PyQt5 import QtWidgets
from configparser import ConfigParser

from utils.thread_evaluation import EvaluateThread
from utils.FSM_ui import TrainingUi

def main():
    # Path to the evaluation directory
    eval_path = '../example'

    # Path to configuration and trained model
    config_file = os.path.join(eval_path, 'config/config.ini')
    model_file = os.path.join(eval_path, 'models/model_120000.zip')
    total_eval_episodes = 10  # Number of evaluation episodes

    # Step 1: Launch the Qt application
    app = QtWidgets.QApplication(sys.argv)
    gui = TrainingUi(config=config_file)
    gui.show()

    # Step 2: Create and start the evaluation thread
    evaluate_thread = EvaluateThread(
        eval_path, config_file, model_file, total_eval_episodes
    )

    # Connect signals to GUI update callbacks
    evaluate_thread.env.action_signal.connect(gui.action_cb)
    evaluate_thread.env.state_signal.connect(gui.state_cb)
    evaluate_thread.env.attitude_signal.connect(gui.attitude_plot_cb)
    evaluate_thread.env.reward_signal.connect(gui.reward_plot_cb)
    evaluate_thread.env.component_signal.connect(gui.component_plot_cb)
    evaluate_thread.env.pose_signal.connect(gui.traj_plot_cb)

    # For special environments, enable additional plot switching
    cfg = ConfigParser()
    cfg.read(config_file)
    env_name = cfg.get('options', 'env_name')
    if env_name == 'Fuse':
        evaluate_thread.env.select_signal.connect(gui.select_plot_cb)

    evaluate_thread.start()

    # Run the Qt main event loop
    sys.exit(app.exec_())


if __name__ == "__main__":
    main()
