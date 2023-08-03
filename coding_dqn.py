# -*- coding: utf-8 -*-
"""
TorchRL trainer: A DQN example
==============================
**Author**: `Vincent Moens <https://github.com/vmoens>`_

"""

##############################################################################
# TorchRL provides a generic :class:`~torchrl.trainers.Trainer` class to handle
# your training loop. The trainer executes a nested loop where the outer loop
# is the data collection and the inner loop consumes this data or some data
# retrieved from the replay buffer to train the model.
# At various points in this training loop, hooks can be attached and executed at
# given intervals.
#
# In this tutorial, we will be using the trainer class to train a DQN algorithm
# to solve the CartPole task from scratch.
#
# Main takeaways:
#
# - Building a trainer with its essential components: data collector, loss
#   module, replay buffer and optimizer.
# - Adding hooks to a trainer, such as loggers, target network updaters and such.
#
# The trainer is fully customisable and offers a large set of functionalities.
# The tutorial is organised around its construction.
# We will be detailing how to build each of the components of the library first,
# and then put the pieces together using the :class:`~torchrl.trainers.Trainer`
# class.
#
# Along the road, we will also focus on some other aspects of the library:
#
# - how to build an environment in TorchRL, including transforms (e.g. data
#   normalization, frame concatenation, resizing and turning to grayscale)
#   and parallel execution. Unlike what we did in the
#   `DDPG tutorial <https://pytorch.org/rl/tutorials/coding_ddpg.html>`_, we
#   will normalize the pixels and not the state vector.
# - how to design a :class:`~torchrl.modules.QValueActor` object, i.e. an actor
#   that estimates the action values and picks up the action with the highest
#   estimated return;
# - how to collect data from your environment efficiently and store them
#   in a replay buffer;
# - how to use multi-step, a simple preprocessing step for off-policy algorithms;
# - and finally how to evaluate your model.
#
# **Prerequisites**: We encourage you to get familiar with torchrl through the
# `PPO tutorial <https://pytorch.org/rl/tutorials/coding_ppo.html>`_ first.
#
# DQN
# ---
#
# DQN (`Deep Q-Learning <https://www.cs.toronto.edu/~vmnih/docs/dqn.pdf>`_) was
# the founding work in deep reinforcement learning.
#
# On a high level, the algorithm is quite simple: Q-learning consists in
# learning a table of state-action values in such a way that, when
# encountering any particular state, we know which action to pick just by
# searching for the one with the highest value. This simple setting
# requires the actions and states to be
# discrete, otherwise a lookup table cannot be built.
#
# DQN uses a neural network that encodes a map from the state-action space to
# a value (scalar) space, which amortizes the cost of storing and exploring all
# the possible state-action combinations: if a state has not been seen in the
# past, we can still pass it in conjunction with the various actions available
# through our neural network and get an interpolated value for each of the
# actions available.
#
# We will solve the classic control problem of the cart pole. From the
# Gymnasium doc from where this environment is retrieved:
#
# | A pole is attached by an un-actuated joint to a cart, which moves along a
# | frictionless track. The pendulum is placed upright on the cart and the goal
# | is to balance the pole by applying forces in the left and right direction
# | on the cart.
#
# .. figure:: /_static/img/cartpole_demo.gif
#    :alt: Cart Pole
#
# We do not aim at giving a SOTA implementation of the algorithm, but rather
# to provide a high-level illustration of TorchRL features in the context
# of this algorithm.

# sphinx_gallery_start_ignore
import tempfile
import warnings

warnings.filterwarnings("ignore")
# sphinx_gallery_end_ignore

import os
import uuid
from tqdm import tqdm

import torch
from torch import nn
from torchrl.collectors import MultiaSyncDataCollector
from torchrl.data import LazyTensorStorage, MultiStep, TensorDictReplayBuffer
from torchrl.envs import (
    EnvCreator,
    ExplorationType,
    ParallelEnv,
    RewardScaling,
    StepCounter,
)
from torchrl.envs.libs.gym import GymEnv
from torchrl.envs.transforms import (
    CatFrames,
    Compose,
    GrayScale,
    ObservationNorm,
    Resize,
    ToTensorImage,
    TransformedEnv,
)
from torchrl.modules import DuelingCnnDQNet, EGreedyWrapper, QValueActor

from torchrl.objectives import DQNLoss, SoftUpdate
from torchrl.record.loggers.csv import CSVLogger




# Let's get started with the various pieces we need for our algorithm:
#
# - An environment;
# - A policy (and related modules that we group under the "model" umbrella);
# - A data collector, which makes the policy play in the environment and
#   delivers training data;
# - A replay buffer to store the training data;
# - A loss module, which computes the objective function to train our policy
#   to maximise the return;
# - An optimizer, which performs parameter updates based on our loss.
#
# Additional modules include a logger, a recorder (executes the policy in
# "eval" mode) and a target network updater. With all these components into
# place, it is easy to see how one could misplace or misuse one component in
# the training script. The trainer is there to orchestrate everything for you!
#
# Building the environment
# ------------------------
#
# First let's write a helper function that will output an environment. As usual,
# the "raw" environment may be too simple to be used in practice and we'll need
# some data transformation to expose its output to the policy.
#
# We will be using five transforms:
#
# - :class:`~torchrl.envs.StepCounter` to count the number of steps in each trajectory;
# - :class:`~torchrl.envs.transforms.ToTensorImage` will convert a ``[W, H, C]`` uint8
#   tensor in a floating point tensor in the ``[0, 1]`` space with shape
#   ``[C, W, H]``;
# - :class:`~torchrl.envs.transforms.RewardScaling` to reduce the scale of the return;
# - :class:`~torchrl.envs.transforms.GrayScale` will turn our image into grayscale;
# - :class:`~torchrl.envs.transforms.Resize` will resize the image in a 64x64 format;
# - :class:`~torchrl.envs.transforms.CatFrames` will concatenate an arbitrary number of
#   successive frames (``N=4``) in a single tensor along the channel dimension.
#   This is useful as a single image does not carry information about the
#   motion of the cartpole. Some memory about past observations and actions
#   is needed, either via a recurrent neural network or using a stack of
#   frames.
# - :class:`~torchrl.envs.transforms.ObservationNorm` which will normalize our observations
#   given some custom summary statistics.
#
# In practice, our environment builder has two arguments:
#
# - ``parallel``: determines whether multiple environments have to be run in
#   parallel. We stack the transforms after the
#   :class:`~torchrl.envs.ParallelEnv` to take advantage
#   of vectorization of the operations on device, although this would
#   technically work with every single environment attached to its own set of
#   transforms.
# - ``obs_norm_sd`` will contain the normalizing constants for
#   the :class:`~torchrl.envs.ObservationNorm` transform.
#


def make_env(
    parallel=False,
    obs_norm_sd=None,
):
    if obs_norm_sd is None:
        obs_norm_sd = {"standard_normal": True}
    base_env = GymEnv(
        "CartPole-v1",
        from_pixels=True,
        pixels_only=True,
        device=device,
    )

    env = TransformedEnv(
        base_env,
        Compose(
            StepCounter(),  # to count the steps of each trajectory
            ToTensorImage(),
            RewardScaling(loc=0.0, scale=0.1),
            GrayScale(),
            Resize(64, 64),
            CatFrames(4, in_keys=["pixels"], dim=-3),
            ObservationNorm(in_keys=["pixels"], **obs_norm_sd),
        ),
    )
    return env



# Compute normalizing constants
#
# To normalize images, we don't want to normalize each pixel independently
# with a full ``[C, W, H]`` normalizing mask, but with simpler ``[C, 1, 1]``
# shaped set of normalizing constants (loc and scale parameters).
# We will be using the ``reduce_dim`` argument
# of :meth:`~torchrl.envs.ObservationNorm.init_stats` to instruct which
# dimensions must be reduced, and the ``keep_dims`` parameter to ensure that
# not all dimensions disappear in the process:
#
def get_norm_stats():
    test_env = make_env()
    test_env.transform[-1].init_stats(
        num_iter=1000, cat_dim=0, reduce_dim=[-1, -2, -4], keep_dims=(-1, -2)
    )
    obs_norm_sd = test_env.transform[-1].state_dict()
    # let's check that normalizing constants have a size of ``[C, 1, 1]`` where
    # ``C=4`` (because of :class:`~torchrl.envs.CatFrames`).
    print("state dict of the observation norm:", obs_norm_sd)
    return obs_norm_sd



# Building the model (Deep Q-network)
# -----------------------------------
#
# The following function builds a :class:`~torchrl.modules.DuelingCnnDQNet`
# object which is a simple CNN followed by a two-layer MLP. The only trick used
# here is that the action values (i.e. left and right action value) are
# computed using
#
# .. math::
#
#    \mathbb{v} = b(obs) + v(obs) - \mathbb{E}[v(obs)]
#
# where :math:`\mathbb{v}` is our vector of action values,
# :math:`b` is a :math:`\mathbb{R}^n \rightarrow 1` function and :math:`v` is a
# :math:`\mathbb{R}^n \rightarrow \mathbb{R}^m` function, for
# :math:`n = \# obs` and :math:`m = \# actions`.
#
# Our network is wrapped in a :class:`~torchrl.modules.QValueActor`,
# which will read the state-action
# values, pick up the one with the maximum value and write all those results
# in the input :class:`tensordict.TensorDict`.
#


def make_model(dummy_env):
    cnn_kwargs = {
        "num_cells": [32, 64, 64],
        "kernel_sizes": [6, 4, 3],
        "strides": [2, 2, 1],
        "activation_class": nn.ELU,
        # This can be used to reduce the size of the last layer of the CNN
        # "squeeze_output": True,
        # "aggregator_class": nn.AdaptiveAvgPool2d,
        # "aggregator_kwargs": {"output_size": (1, 1)},
    }
    mlp_kwargs = {
        "depth": 2,
        "num_cells": [
            64,
            64,
        ],
        "activation_class": nn.ELU,
    }
    net = DuelingCnnDQNet(
        dummy_env.action_spec.shape[-1], 1, cnn_kwargs, mlp_kwargs
    ).to(device)
    net.value[-1].bias.data.fill_(init_bias)

    actor = QValueActor(net, in_keys=["pixels"], spec=dummy_env.action_spec).to(device)
    # init actor: because the model is composed of lazy conv/linear layers,
    # we must pass a fake batch of data through it to instantiate them.
    tensordict = dummy_env.fake_tensordict()
    actor(tensordict)

    # we wrap our actor in an EGreedyWrapper for data collection
    actor_explore = EGreedyWrapper(
        actor,
        annealing_num_steps=total_frames,
        eps_init=eps_greedy_val,
        eps_end=eps_greedy_val_env,
    )

    return actor, actor_explore



# Collecting and storing data
# ---------------------------
#
# Replay buffers
# ~~~~~~~~~~~~~~
#
# Replay buffers play a central role in off-policy RL algorithms such as DQN.
# They constitute the dataset we will be sampling from during training.
#
# Here, we will use a regular sampling strategy, although a prioritized RB
# could improve the performance significantly.
#
# We place the storage on disk using
# :class:`~torchrl.data.replay_buffers.storages.LazyMemmapStorage` class. This
# storage is created in a lazy manner: it will only be instantiated once the
# first batch of data is passed to it.
#
# The only requirement of this storage is that the data passed to it at write
# time must always have the same shape.


def get_replay_buffer(buffer_size, batch_size):
    replay_buffer = TensorDictReplayBuffer(
        batch_size=batch_size,
        storage=LazyTensorStorage(max_size=buffer_size, device=device),
    )
    return replay_buffer



# Loss function
# -------------
#
# Building our loss function is straightforward: we only need to provide
# the model and a bunch of hyperparameters to the DQNLoss class.
#
# Target parameters
#
# Many off-policy RL algorithms use the concept of "target parameters" when it
# comes to estimate the value of the next state or state-action pair.
# The target parameters are lagged copies of the model parameters. Because
# their predictions mismatch those of the current model configuration, they
# help learning by putting a pessimistic bound on the value being estimated.
# This is a powerful trick (known as "Double Q-Learning") that is ubiquitous
# in similar algorithms.
#


def get_loss_module(actor, gamma):
    loss_module = DQNLoss(actor, delay_value=True)
    loss_module.make_value_estimator(gamma=gamma)
    target_updater = SoftUpdate(loss_module, eps=0.995)
    return loss_module, target_updater



# Hyperparameters
# ---------------
#
# Let's start with our hyperparameters. The following setting should work well
# in practice, and the performance of the algorithm should hopefully not be
# too sensitive to slight variations of these.
device = "cuda:0" if torch.cuda.device_count() > 0 else "cpu"

# Optimizer parameters
# the learning rate of the optimizer
lr = 2e-3
# weight decay
wd = 1e-5
# the beta parameters of Adam
betas = (0.9, 0.999)

# DQN parameters
# gamma decay factor
gamma = 0.99

# Smooth target network update decay parameter.
# This loosely corresponds to a 1/tau interval with hard target network
# update
tau = 0.02


# Replay buffer
#
# Total frames collected in the environment. In other implementations, the
# user defines a maximum number of episodes.
total_frames = 500000
# Random frames used to initialize the replay buffer.
init_random_frames = 1000
# Frames sampled from the replay buffer at each optimization step
batch_size = 256
# Size of the replay buffer in terms of frames
buffer_size = min(total_frames, 100000)


# Environment and exploration
#
# We set the initial and final value of the epsilon factor in Epsilon-greedy
# exploration.
# Since our policy is deterministic, exploration is crucial: without it, the
# only source of randomness would be the environment reset.
eps_greedy_val = 0.1
eps_greedy_val_env = 0.005
# To speed up learning, we set the bias of the last layer of our value network
# to a predefined value (this is not mandatory)
init_bias = 2.0



# Prepare for training
# ---------------
# Normalize environment frames
stats = get_norm_stats()
test_env = make_env(parallel=False, obs_norm_sd=stats)
# Get model
actor, actor_explore = make_model(test_env)
loss_module, target_net_updater = get_loss_module(actor, gamma)
optimizer = torch.optim.Adam(
    loss_module.parameters(), lr=lr, weight_decay=wd, betas=betas
)
exp_name = f"dqn_exp_{uuid.uuid1()}"
tmpdir = tempfile.TemporaryDirectory()
logger = CSVLogger(exp_name=exp_name, log_dir=tmpdir.name)
warnings.warn(f"log dir: {logger.experiment.log_dir}")



# Training loop
# ------------- 
# first initialize the replay buffer with greedy exploration
rb = get_replay_buffer(buffer_size, batch_size)
for frame in tqdm(range(init_random_frames)):
    td = test_env.reset()
    td = actor_explore(td.clone())
    td = test_env.step(td)
    rb.add(td.clone())

td = test_env.reset()
for frame in tqdm(range(total_frames)):
    td = actor(td.clone())
    rb.add(td.clone())

    td_batch = rb.sample()
    loss_td = loss_module(td_batch)
    loss = loss_td["loss"]
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    target_net_updater.step()
    mean_batch_reward = td_batch.get(("next", "reward")).mean()
    if frame % 1000 == 0:
        print(f"At step {frame}, loss: {loss.item():.3f}, mean batch reward: {mean_batch_reward}")


