from gym.envs.registration import register

register(
    id='2ArmBandit-v0',
    entry_point='gym_envs.envs:mabEnv',
) 
register(
    id='10ArmGaussian-v0',
    entry_point='gym_envs.envs:GaussianEnv',
)
register(
    id='RandomWalk-v0',
    entry_point='gym_envs.envs:RandomWalkEnv',
)