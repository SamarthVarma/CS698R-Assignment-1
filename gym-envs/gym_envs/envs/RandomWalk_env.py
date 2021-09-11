import gym
from gym import error, spaces, utils
from gym.utils import seeding
import numpy as np

class RandomWalkEnv(gym.Env):
    def __init__(self):
        self.observation_space = spaces.Discrete(5)
        self.action_space = spaces.Discrete(2)
        self.state = 3
        self.seed()

    def seed(self ,seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    def step(self, action):
        assert action == 0 or action == 1
        reward = 0
        intended_step = np.random.binomial(1,0.5)
        if intended_step == 1:
            self.state = self.state - 1 + 2 *action 
        else:
             self.state = self.state + 1 - 2 * action
        done = False
        if(self.state == 0): done = True
        if(self.state == 6): 
            done = True
            reward = 1 

        return self.state, reward, done, {}
    
    def reset(self):
        self.state = 3
        return self.state, False

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