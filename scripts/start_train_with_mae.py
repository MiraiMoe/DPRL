import os
from utils.multi_env_thread_train import TrainingThread

def main():
    # set config
    base_dir = os.path.dirname(os.path.abspath(__file__))
    config_path = os.path.join(base_dir, '..', 'configs', 'config_new.ini')

    # Initialize Training Thread
    training_thread = TrainingThread(config_path)

    # Start training thread
    training_thread.run()

    print('Exiting program')


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print('system exit')
