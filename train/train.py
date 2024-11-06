import sys

from frame.config_handle import version_controlled_execution_context
from frame.submit import submit_jobs
from train.train_config import TrainConfig

def main(argv):
    config = TrainConfig.load_from_files(argv)
    with version_controlled_execution_context(config) as context:
        submit_jobs(config)  # todo: how to differentiate when running train from other stuff here?

if __name__ == "__main__":
    # argparse and stuff

    # Call main with configuration paths
    main(sys.argv[1:])
