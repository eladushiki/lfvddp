from train.train_config import TrainConfig


def submit_jobs(train_config: TrainConfig) -> None:
    for job in range(train_config.cluster_config.njobs):
        submit_job(train_config)

def submit_job(train_config: TrainConfig) -> None:
    pass
