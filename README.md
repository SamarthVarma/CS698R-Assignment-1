# CS698R-Assignment-1
Code of Assignment 1 of Deep Reinforcement Learning as attached. Basics of Multi Armed Bandits, Monte Carlo Estimation and TD Learning.


* Custom Environments
* Question 1
* Question 2

<h2> Custom Environments </h2>

* There are three openai gym environments in the gym-envs folder, for 2 Armed Bernoulli Bandit, 10 Armed Gaussian Bandit and Random Walk Environment with 5 States and 2 Terminal States.

* The gym id's for the environments are '2ArmBandit-v0' for 2 Armed Bernoulli Bandit, '10ArmGaussian-v0' for 10 Armed Gaussian Bandit and 'RandomWalk-v0' for Random Walk Environment.

<h4> DoubleArmBernoulliBandit_env.py (2 Armed Bernoulli Bandit) </h4>

* It takes $\alpha$ and $\beta$ as the parameters and the action space is discrete(2)

* The class mabEnv wraps the gym.env API

<h4> RandomWalk_env.py (2 Armed Bernoulli Bandit) </h4>

* It takes no of states (including terminal) as the parameter (default = 7). It starts at state (total states/2) and the action space is discrete(2)

* The class RandomWalkEnv wraps the gym.env API

<h4> TenArmGaussianBandit_env.py (2 Armed Bernoulli Bandit) </h4>

* It takes the standard deviation as the parameters (default = 1) and the action space is discrete(10) 

* The class GaussianEnv wraps the gym.env API