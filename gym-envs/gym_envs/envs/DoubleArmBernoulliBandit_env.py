import gym
from gym import error, spaces, utils
from gym.utils import seeding
import numpy as np

class mabEnv(gym.Env):
    def __init__(self, alpha, beta):
        self.observation_space = spaces.Discrete(1)
        self.action_space = spaces.Discrete(2)
        self.state = None
        self.alpha = alpha
        self.beta = beta
        self.prob = [self.alpha, self.beta]
        self.seed()

    def seed(self ,seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    def step(self, action):
        assert action == 0 or action == 1
        reward = 0
        reward = np.random.binomial(1,self.prob[action])
        return 0, reward, True, {}
    
    def reset(self):
        return 0

    def close(self):
        pass

    def render(self, mode='human', close=False):
        pass



"""     ...
  def seed
  def step(self, action):
    ...
  def reset(self):
    ...
  def close(self):
    ... """