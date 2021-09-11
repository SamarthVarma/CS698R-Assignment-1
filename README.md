# CS698R-Assignment-1
Code of Assignment 1 of Deep Reinforcement Learning as attached. Basics of Multi Armed Bandits, Monte Carlo Estimation and TD Learning.


* Custom Environments
* Question 1
* Question 2

<h2> Custom Environments </h2>

* There are three openai gym environments in the gym-envs folder, for 2 Armed Bernoulli Bandit, 10 Armed Gaussian Bandit and Random Walk Environment with 5 States and 2 Terminal States.

* The gym id's for the environments are '2ArmBandit-v0' for 2 Armed Bernoulli Bandit, '10ArmGaussian-v0' for 10 Armed Gaussian Bandit and 'RandomWalk-v0' for Random Walk Environment.

<h4> DoubleArmBernoulliBandit_env.py (2 Armed Bernoulli Bandit) </h4>

* It takes α and β as the parameters and the action space is discrete(2)

* The class mabEnv wraps the gym.env API

<h4> RandomWalk_env.py (2 Armed Bernoulli Bandit) </h4>

* It takes no of states (including terminal) as the parameter (default = 7). It starts at state (total states/2) and the action space is discrete(2)

* The class RandomWalkEnv wraps the gym.env API

<h4> TenArmGaussianBandit_env.py (2 Armed Bernoulli Bandit) </h4>

* It takes the standard deviation as the parameters (default = 1) and the action space is discrete(10) 

* The class GaussianEnv wraps the gym.env API

<h2> Question 1 </h2>

<h4> All Questions are solved in the function declared as question number. Example, Question 5 is solved in the function declared as q5(), Question 9 as q9(). This is True for Questions 4-9. They take no parameter as input</h4>

<h4> The seed taken for numpy and the environment is 373. For 50 different α and β, the seeds are declared as follows:
    
    np.random.seed(373)
    alpha = np.random.uniform(size=50)
    beta = np.random.uniform(size=50)

    OR

    np.random.seed(373)
    SD = np.random.uniform(size=50)  

For each alpha[i], beta[i]; the environments are declared with seeds 373 + i. 

    for i in range(50):    
        env[i] = gym.make('2ArmBandit-v0', alpha = alpha[i], beta = beta[i])
        env[i].seed(373+i)
        np.random.seed(373+i)
        env[i].action_space.seed(373+i)
</h4>

<h2> Question 1 </h2>
solved under function 'doublearmbernoullibandit(hyperparameters, ep_count)'. An example of Parameters would be 
        hyperparameters = {
            'alpha': [0,1,0,1,0.5],
            'beta': [0,0,1,1,0.5]
            'epsilon':0.4
        }
        episode_count = 10000 

<h2> Question 2 </h2>
solved under function 'TenArmGaussian(hyperparameters, ep_count)'. An example of Parameters would be 
        hyperparameters = {
            'sd': [0,0.2,0.4,0.5,0.75,1,50,100]
        }
        episode_count = 1000

<h2> Question 3 </h2>
<h1> Part I : Pure Exploitation </h1>
Defined under the function 'pure_exploitation(env, ep_count)'. Takes a seeded environment as the parameter and the number of episodes.
<h1> Part II : Pure Exploration </h1>
Defined under the function 'pure_exploration(env, ep_count)'. Takes a seeded environment as the parameter and the number of episodes.
<h1> Part III : Epsilon Greedy </h1>
Defined under the function 'epsilon_greedy(env, ep_count, epsilon = 0.5)'. Takes a seeded environment as the parameter, number of episodes and the value of epsilon.
<h1> Part VI : Epsilon Decay </h1>
Defined under the function 'epsilon_decay(env, ep_count, decay_type, final_rate = 0.01)'. Takes a seeded environment as the parameter, number of episodes, decay_type ('linear' or 'exponential') and final_rate which is defaulted to 0.01
<h1> Part V : Softmax </h1>
Defined under the function 'softmax(env, ep_count, temp)'. Takes a seeded environment as the parameter, number of episodes and the temperature (tau)
<h1> Part VI : UCB </h1>
Defined under the function 'UCB(env, ep_count, c)'. Takes a seeded environment as the parameter, the number of episodes and the constant C.

All the Training Algorithms return q values, action taken by the agent and the reward for all the episodes.

<h2> Question 4 </h2>

As defined in function 'q4()'. It takes no parameters as arguments.

The various parameters for the algorithms are:

    epsilon = 0.5
    steps = 1000
    decay_type = 'linear'
    c = 0.2
    temp = 100

p_exploitation, p_exploration,e_greedy,e_decay,p_softmax, p_ucb are n-dimensional arrays corresponding to the rewards of each 50 environments. They average out for the 50 environments in the array means.

<h2> Question 5 </h2>

As defined in function 'q5()'. It takes no parameters as arguments.

The logic and the variables are the same as q4() except that the environment is different.


<h2> Question 6 </h2>

As defined in function 'q6()'. It takes no parameters as arguments.

The various parameters for the algorithms are:

    epsilon = 0.5
    steps = 1000
    decay_type = 'linear'
    c = 0.2
    temp = 100

q and action are n-dimensional arrays corresponding to the q_est and action taken of each 50 environments. 
The array 'v' is the optimal action corresponding to the final q value. Tot_regret is final regret averaged out over regrets of the 50 environments.

<h2> Question 7 </h2>

As defined in function 'q7()'. It takes no parameters as arguments.

The logic and the variables are the same as q6() except that the environment is different.

<h2> Question 8 </h2>

As defined in function 'q8()'. It takes no parameters as arguments.

The various parameters for the algorithms are:

    epsilon = 0.5
    steps = 1000
    decay_type = 'linear'
    c = 0.2
    temp = 100

p_exploitation, p_exploration,e_greedy,e_decay,p_softmax, p_ucb are n-dimensional arrays corresponding to the array of tuples of q_est, action taken and rewards of each 50 environments. 
p_exploitation[i][1] is the array of action taken for i-th environment for pure exploitation while qmax is the argmax of final q value (optimal action) corresponding to each algorithms (There are 6 Algorithms)

<h2> Question 9 </h2>

As defined in function 'q9()'. It takes no parameters as arguments.

The logic and the variables are the same as q8() except that the environment is different.

