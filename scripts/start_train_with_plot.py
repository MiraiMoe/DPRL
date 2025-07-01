import sys
import os
from PyQt5 import QtWidgets
from utils.thread_train import TrainingThread
from utils.FSM_ui import TrainingUi

def main():
    # set config
    base_dir = os.path.dirname(os.path.abspath(__file__))
    config_path = os.path.join(base_dir, '..', 'configs', 'config_new.ini')

    # Initialize Qt App
    app = QtWidgets.QApplication(sys.argv)
    gui = TrainingUi(config_path)
    gui.show()

    # Initialize Training Thread
    training_thread = TrainingThread(config_path)

    # Connect signals from environment to GUI callbacks
    training_thread.env.action_signal.connect(gui.action_cb)
    training_thread.env.state_signal.connect(gui.state_cb)
    training_thread.env.attitude_signal.connect(gui.attitude_plot_cb)
    training_thread.env.reward_signal.connect(gui.reward_plot_cb)
    training_thread.env.component_signal.connect(gui.component_plot_cb)
    training_thread.env.pose_signal.connect(gui.traj_plot_cb)

    # Start training thread
    training_thread.start()

    # Execute GUI loop
    sys.exit(app.exec_())
    print('Exiting program')


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print('system exit')
