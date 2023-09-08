import argparse
import os
from typing import Callable, Optional, Tuple

from torch.utils.tensorboard import SummaryWriter

from tianshou.utils import TensorboardLogger
from tianshou.utils.logger.base import LOG_DATA_TYPE

import wandb



class WandbLogger:
    """ @TODO Cite the original source
    Adpated from TianShou's WandbLogger class to log to wandb.ai
    Weights and Biases logger that sends data to https://wandb.ai/.

    This logger creates three panels with plots: train, test, and update.
    Make sure to select the correct access for each panel in weights and biases:

    Example of usage:
    ::

        logger = WandbLogger()
        logger.load(SummaryWriter(log_path))
        result = onpolicy_trainer(policy, train_collector, test_collector,
                                  logger=logger)

    :param int train_interval: the log interval in log_train_data(). Default to 1000.
    :param int test_interval: the log interval in log_test_data(). Default to 1.
    :param int update_interval: the log interval in log_update_data().
        Default to 1000.
    :param int save_interval: the save interval in save_data(). Default to 1 (save at
        the end of each epoch).
    :param bool write_flush: whether to flush tensorboard result after each
        add_scalar operation. Default to True.
    :param str project: W&B project name. Default to "tianshou".
    :param str name: W&B run name. Default to None. If None, random name is assigned.
    :param str entity: W&B team/organization name. Default to None.
    :param str run_id: run id of W&B run to be resumed. Default to None.
    :param argparse.Namespace config: experiment configurations. Default to None.
    """

    def __init__(
        self,
        train_interval: int = 1000,
        test_interval: int = 1,
        update_interval: int = 1000,
        save_interval: int = 1000,
        write_flush: bool = True,
        project: Optional[str] = None,
        name: Optional[str] = None,
        entity: Optional[str] = None,
        run_id: Optional[str] = None,
        config: Optional[argparse.Namespace] = None,
    ) -> None:
        self.train_interval = train_interval
        self.test_interval = test_interval
        self.update_interval = update_interval
        self.save_interval = save_interval
        self.write_flush = write_flush
        self.last_log_train_step = -1
        self.last_log_test_step = -1
        self.last_log_update_step = -1
        self.last_save_step = -1        
        self.restored = False        
        if project is None:
            project = os.getenv("WANDB_PROJECT", "tianshou")

        self.wandb_run = wandb.init(
            project=project,
            name=name,
            id=run_id,
            resume="allow",
            entity=entity,
            sync_tensorboard=True,
            monitor_gym=True,
            config=config,  # type: ignore
        ) if not wandb.run else wandb.run
        self.wandb_run._label(repo="tianshou")  # type: ignore
        self.tensorboard_logger: Optional[TensorboardLogger] = None

    def load(self, writer: SummaryWriter) -> None:
        self.writer = writer
        self.tensorboard_logger = TensorboardLogger(
            writer, self.train_interval, self.test_interval, self.update_interval,
            self.save_interval, self.write_flush
        )

    def write(self, step_type: str, step: int, data: LOG_DATA_TYPE) -> None:
        if self.tensorboard_logger is None:
            raise Exception(
                "`logger` needs to load the Tensorboard Writer before "
                "writing data. Try `logger.load(SummaryWriter(log_path))`"
            )
        else:
            self.tensorboard_logger.write(step_type, step, data)

    def log_train_data(self, collect_result: dict, step: int) -> None:
        """Log statistics generated during training.

        :param collect_result: a dict containing information of data collected in
            training stage, i.e., returns of collector.collect().
        :param int step: stands for the timestep the collect_result being logged.
        """
        if collect_result["n/ep"] > 0:
            if step - self.last_log_train_step >= self.train_interval:
                log_data = {
                    "train/episode": collect_result["n/ep"],
                    "train/reward": collect_result["rew"],
                    "train/length": collect_result["len"],
                }
                self.write("train/env_step", step, log_data)
                self.last_log_train_step = step
    
    def log_test_data(self, collect_result: dict, step: int) -> None:
        """Use writer to log statistics generated during evaluating.

        :param collect_result: a dict containing information of data collected in
            evaluating stage, i.e., returns of collector.collect().
        :param int step: stands for the timestep the collect_result being logged.
        """
        assert collect_result["n/ep"] > 0
        if step - self.last_log_test_step >= self.test_interval:            
            log_data = {
                "test/env_step": step,
                "test/reward": collect_result["rew"],
                "test/length": collect_result["len"],
                "test/reward_std": collect_result["rew_std"],
                "test/length_std": collect_result["len_std"],
            }
            self.write("test/env_step", step, log_data)
            self.last_log_test_step = step

    def log_update_data(self, update_result: dict, step: int) -> None:
        """Use writer to log statistics generated during updating.

        :param update_result: a dict containing information of data collected in
            updating stage, i.e., returns of policy.update().
        :param int step: stands for the timestep the collect_result being logged.
        """
        if step - self.last_log_update_step >= self.update_interval:
            log_data = {f"update/{k}": v for k, v in update_result.items()}
            self.write("update/gradient_step", step, log_data)
            self.last_log_update_step = step

    def save_data(
        self,
        epoch: int,
        env_step: int,
        gradient_step: int,
        save_checkpoint_fn: Optional[Callable[[int, int, int], str]] = None,
    ) -> None:
        """Use writer to log metadata when calling ``save_checkpoint_fn`` in trainer.

        :param int epoch: the epoch in trainer.
        :param int env_step: the env_step in trainer.
        :param int gradient_step: the gradient_step in trainer.
        :param function save_checkpoint_fn: a hook defined by user, see trainer
            documentation for detail.
        """
        if save_checkpoint_fn and epoch - self.last_save_step >= self.save_interval:
            self.last_save_step = epoch
            checkpoint_path = save_checkpoint_fn(epoch, env_step, gradient_step)

            checkpoint_artifact = wandb.Artifact(
                'run_' + self.wandb_run.id + '_checkpoint',  # type: ignore
                type='model',
                metadata={
                    "save/epoch": epoch,
                    "save/env_step": env_step,
                    "save/gradient_step": gradient_step,
                    "checkpoint_path": str(checkpoint_path),
                }
            )
            checkpoint_artifact.add_file(str(checkpoint_path))
            self.wandb_run.log_artifact(checkpoint_artifact)  # type: ignore

    def restore_data(self) -> Tuple[int, int, int]:
        checkpoint_artifact = self.wandb_run.use_artifact(    # type: ignore
            f"run_{self.wandb_run.id}_checkpoint:latest"  # type: ignore
        )
        assert checkpoint_artifact is not None, "W&B dataset artifact doesn't exist"

        checkpoint_artifact.download(
            os.path.dirname(checkpoint_artifact.metadata['checkpoint_path'])
        )

        try:  # epoch / gradient_step
            epoch = checkpoint_artifact.metadata["save/epoch"]
            self.last_save_step = self.last_log_test_step = epoch
            gradient_step = checkpoint_artifact.metadata["save/gradient_step"]
            self.last_log_update_step = gradient_step
        except KeyError:
            epoch, gradient_step = 0, 0
        try:  # offline trainer doesn't have env_step
            env_step = checkpoint_artifact.metadata["save/env_step"]
            self.last_log_train_step = env_step
        except KeyError:
            env_step = 0
        return epoch, env_step, gradient_step
