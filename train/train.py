from argparse import ArgumentParser
from pathlib import Path

from frame.config_handle import version_controlled_execution_context
from frame.submit import submit_jobs
from train.train_config import TrainConfig

def train(cluster_config_path: Path, train_config_path: Path) -> None:
    config = TrainConfig.load_from_files([cluster_config_path, train_config_path])
    with version_controlled_execution_context(config) as context:
        submit_jobs(config)  # todo: how to differentiate when running train from other stuff here?

def main():
    parser = ArgumentParser()

    # Arguments for running training
    parser.add_argument(
        "--cluster-config", type=Path, required=True,
        help="Path to cluster configuration file", dest="cluster_config_path"
    )
    parser.add_argument(
        "--train-config", type=Path, required=True,
        help="Path to training configuration file", dest="train_config_path"
    )

    args = parser.parse_args()

    # Call main with configuration paths
    train(args.cluster_config_path, args.train_config_path)

if __name__ == "__main__":
    main()
