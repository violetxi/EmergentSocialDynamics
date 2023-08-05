from marllib import marl


# prepare env
env = marl.make_env(environment_name="mpe", map_name="simple_spread")

# initialize algorithm with appointed hyper-parameters
mappo = marl.algos.mappo(hyperparam_source="mpe")

# build agent model based on env + algorithms + user preference
model = marl.build_model(env, mappo, {"core_arch": "mlp", "encode_layer": "128-256"})

# start training
mappo.fit(env, model, stop={'episode_reward_mean': 2000, 'timesteps_total': 10000000}, local_mode=False, num_gpus=1,
          num_workers=1, share_policy='group', checkpoint_freq=50)