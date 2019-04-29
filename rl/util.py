import logging

from baselines.common.atari_wrappers import TransitionEnvWrapper


def make_env(env_name, config=None):
    import gym
    env = gym.make(env_name)
    gym.logger.setLevel(logging.WARN)
    if config:
        try:
            env.unwrapped.set_environment_config(config.env_args)
            gym.logger.info("Set the configuration to the environment: "
                            "{}".format(config.env_args))
        except:
            gym.logger.info("Can't set the configuration to the environment! "
                            "Use the default setting instead of "
                            "({})".format(config.env_args))

        assert env.spec.max_episode_steps <= config.num_rollouts, \
            '--num_rollouts ({}) should be larger than a game length ({})'.format(
                config.num_rollouts, env.spec.max_episode_steps)

    env = TransitionEnvWrapper(env)
    return env

