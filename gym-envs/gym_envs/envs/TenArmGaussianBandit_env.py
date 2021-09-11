import gym
from gym import error, spaces, utils
from gym.utils import seeding
import numpy as np

class GaussianEnv(gym.Env):
    def __init__(self, sd = 1):
        self.seed()
        self.observation_space = spaces.Discrete(1)
        self.action_space = spaces.Discrete(10)
        self.state = None
        self.sd = sd
        self.q = np.random.normal(0, self.sd, 10)

    def seed(self ,seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    def step(self, action):
        assert action < 10 and action >= 0
       # print(self.q)
        reward = np.random.normal(self.q[action], self.sd)
        return 0, reward, True, {}

    def close(self):
        pass

    def reset(self):
        self.q = np.random.normal(0, self.sd, 10)
        return self.q

    def render(self, mode='human', close=False):
        pass