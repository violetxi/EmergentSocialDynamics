import time
import tqdm
from torch.utils.tensorboard import SummaryWriter
from typing import Dict, List, Union, Callable, Optional

from tianshou_elign.data import Collector
from tianshou_elign.policy import BasePolicy
from tianshou_elign.utils import tqdm_config, MovAvg
from tianshou_elign.trainer import test_episode, gather_info


def offpolicy_trainer(
        policy: BasePolicy,
        train_collector: Collector,
        test_collector: Collector,
        max_epoch: int,
        step_per_epoch: int,
        collect_per_step: int,
        episode_per_test: Union[int, List[int]],
        batch_size: int,
        train_fn: Optional[Callable[[int], None]] = None,
        test_fn: Optional[Callable[[int], None]] = None,
        stop_fn: Optional[Callable[[float], bool]] = None,
        save_fn: Optional[Callable[[BasePolicy], None]] = None,
        save_per_epoch: Optional[int] = None,
        log_fn: Optional[Callable[[dict], None]] = None,
        writer: Optional[SummaryWriter] = None,
        log_interval: int = 1,
        verbose: bool = True,
        **kwargs
) -> Dict[str, Union[float, str]]:
    """A wrapper for off-policy trainer procedure.

    :param policy: an instance of the :class:`~tianshou.policy.BasePolicy`
        class.
    :param train_collector: the collector used for training.
    :type train_collector: :class:`~tianshou.data.Collector`
    :param test_collector: the collector used for testing.
    :type test_collector: :class:`~tianshou.data.Collector`
    :param int max_epoch: the maximum of epochs for training. The training
        process might be finished before reaching the ``max_epoch``.
    :param int step_per_epoch: the number of step for updating policy network
        in one epoch.
    :param int collect_per_step: the number of frames the collector would
        collect before the network update. In other words, collect some frames
        and do one policy network update.
    :param episode_per_test: the number of episodes for one policy evaluation.
    :param int batch_size: the batch size of sample data, which is going to
        feed in the policy network.
    :param function train_fn: a function receives the current number of epoch
        index and performs some operations at the beginning of training in this
        epoch.
    :param function test_fn: a function receives the current number of epoch
        index and performs some operations at the beginning of testing in this
        epoch.
    :param function save_fn: a function for saving policy when the undiscounted
        average mean reward in evaluation phase gets better.
    :param function stop_fn: a function receives the average undiscounted
        returns of the testing result, return a boolean which indicates whether
        reaching the goal.
    :param function log_fn: a function receives env info for logging.
    :param torch.utils.tensorboard.SummaryWriter writer: a TensorBoard
        SummaryWriter.
    :param int log_interval: the log interval of the writer.
    :param bool verbose: whether to print the information.

    :return: See :func:`~tianshou.trainer.gather_info`.
    """
    global_step = 0
    best_epoch, best_reward = -1, -1
    stat = {}
    start_time = time.time()
    test_in_train = train_collector.policy == policy
    for epoch in range(1, 1 + max_epoch):
        # train
        policy.train()
        if train_fn:
            train_fn(epoch)
        with tqdm.tqdm(total=step_per_epoch, desc=f'Epoch #{epoch}',
                       **tqdm_config) as t:
            while t.n < t.total:
                result = train_collector.collect(n_step=collect_per_step,
                                                 log_fn=log_fn)
                data = {}
                if test_in_train and stop_fn and stop_fn(result['rew']):
                    test_result = test_episode(
                        policy, test_collector, test_fn,
                        epoch, episode_per_test)
                    if stop_fn and stop_fn(test_result['rew']):
                        if save_fn:
                            # save_fn(policy)
                            save_fn()
                        for k in result.keys():
                            data[k] = f'{result[k]:.2f}'
                        t.set_postfix(**data)
                        return gather_info(
                            start_time, train_collector, test_collector,
                            test_result['rew'])
                    else:
                        policy.train()
                        if train_fn:
                            train_fn(epoch)
                for i in range(min(
                        result['n/st'] // collect_per_step, t.total - t.n)):
                    global_step += 1
                    sample_batch, intr_rews, _ = train_collector.sample(batch_size)
                    losses = policy.learn(sample_batch)
                    for k in result.keys():
                        data[k] = f'{result[k]:.2f}'
                        if writer and global_step % log_interval == 0:
                            writer.add_scalar(
                                k, result[k], global_step=global_step)
                    for k in losses.keys():
                        if stat.get(k) is None:
                            stat[k] = MovAvg()
                        stat[k].add(losses[k])
                        data[k] = f'{stat[k].get():.6f}'
                        if writer and global_step % log_interval == 0:
                            writer.add_scalar(
                                k, stat[k].get(), global_step=global_step)
                    for k in intr_rews.keys():
                        data[k] = f'{intr_rews[k]:.2f}'
                        if writer and global_step % log_interval == 0:
                            writer.add_scalar(
                                k, intr_rews[k], global_step=global_step)                            
                    t.update(1)
                    t.set_postfix(**data)
            if t.n <= t.total:
                t.update()
        # test
        result = test_episode(
            policy, test_collector, test_fn, epoch, episode_per_test)
        for k in result.keys():
            if writer:
                writer.add_scalar(
                    'test_' + k, result[k], global_step=epoch)
       
        if best_epoch == -1 or best_reward < result['rew']:
            best_reward = result['rew']
            best_epoch = epoch
            if writer:
                if 'loss/world_model' in losses:
                    writer.add_scalar(
                        'wm_loss', losses['loss/world_model'], global_step=epoch)
                writer.add_scalar(
                    'best_epoch', best_epoch, global_step=epoch)
                writer.add_scalar(
                    'best_rew', best_reward, global_step=epoch)
            if save_fn:
                # save_fn(policy)
                save_fn()
        if save_per_epoch is not None:
            if save_per_epoch % epoch == 0:
                # save_fn(policy)
                save_fn()
        if verbose:
            print(f'Epoch #{epoch}: test_reward: {result["rew"]:.6f}, '
                  f'best_reward: {best_reward:.6f} in #{best_epoch}')
        # if epoch - best_epoch >= 100:
        #     print("Early stopping when the best eval reward has not changed after 100 epochs.")
        #     break
        if stop_fn and stop_fn(best_reward):
            break
    return gather_info(
        start_time, train_collector, test_collector, best_reward)
