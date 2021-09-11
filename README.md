# CS698R-Assignment-1
Code of Assignment 1 of Deep Reinforcement Learning as attached. Basics of Multi Armed Bandits, Monte Carlo Estimation and TD Learning.


* [Custom Environments](#custom1)
* [Question 1](#question1)
* [Question 2](#question2)


<h2 name = "custom1">Custom Environments</h2>

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

<h2 name = "question1"> Question 1 (train.py) </h2>

*All Questions are solved in the function declared as question number. Example, Question 5 is solved in the function declared as q5(), Question 9 as q9(). This is True for Questions 4-9. They take no parameter as input

*The seed taken for numpy and the environment is 373. For 50 different α and β, the seeds are declared as follows:
    
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
        env[i].reset()


<h3> SubQuestion 1 </h3>
solved under function 'doublearmbernoullibandit(hyperparameters, ep_count)'. An example of Parameters would be 
        hyperparameters = {
            'alpha': [0,1,0,1,0.5],
            'beta': [0,0,1,1,0.5]
            'epsilon':0.4
        }
        episode_count = 10000 

<h3> SubQuestion 2 </h3>
solved under function 'TenArmGaussian(hyperparameters, ep_count)'. An example of Parameters would be 
        hyperparameters = {
            'sd': [0,0.2,0.4,0.5,0.75,1,50,100]
        }
        episode_count = 1000

<h3> SubQuestion 3 </h3>
<h4> Part I : Pure Exploitation </h4>
Defined under the function 'pure_exploitation(env, ep_count)'. Takes a seeded environment as the parameter and the number of episodes.
<h4> Part II : Pure Exploration </h4>
Defined under the function 'pure_exploration(env, ep_count)'. Takes a seeded environment as the parameter and the number of episodes.
<h4> Part III : Epsilon Greedy </h4>
Defined under the function 'epsilon_greedy(env, ep_count, epsilon = 0.5)'. Takes a seeded environment as the parameter, number of episodes and the value of epsilon.
<h4> Part VI : Epsilon Decay </h4>
Defined under the function 'epsilon_decay(env, ep_count, decay_type, final_rate = 0.01)'. Takes a seeded environment as the parameter, number of episodes, decay_type ('linear' or 'exponential') and final_rate which is defaulted to 0.01
<h4> Part V : Softmax </h4>
Defined under the function 'softmax(env, ep_count, temp)'. Takes a seeded environment as the parameter, number of episodes and the temperature (tau)
<h4> Part VI : UCB </h4>
Defined under the function 'UCB(env, ep_count, c)'. Takes a seeded environment as the parameter, the number of episodes and the constant C.

All the Training Algorithms return q values, action taken by the agent and the reward for all the episodes.

<h3> SubQuestion 4 </h3>

As defined in function 'q4()'. It takes no parameters as arguments.

The various parameters for the algorithms are:

    epsilon = 0.5
    steps = 1000
    decay_type = 'linear'
    c = 0.2
    temp = 100

p_exploitation, p_exploration,e_greedy,e_decay,p_softmax, p_ucb are n-dimensional arrays corresponding to the rewards of each 50 environments. They average out for the 50 environments in the array means.

<h3> SubQuestion 5 </h3>

As defined in function 'q5()'. It takes no parameters as arguments.

The logic and the variables are the same as q4() except that the environment is different.


<h3> SubQuestion 6 </h3>

As defined in function 'q6()'. It takes no parameters as arguments.

The various parameters for the algorithms are:

    epsilon = 0.5
    steps = 1000
    decay_type = 'linear'
    c = 0.2
    temp = 100

q and action are n-dimensional arrays corresponding to the q_est and action taken of each 50 environments. 
The array 'v' is the optimal action corresponding to the final q value. Tot_regret is final regret averaged out over regrets of the 50 environments.

<h3> SubQuestion 7 </h3>

As defined in function 'q7()'. It takes no parameters as arguments.

The logic and the variables are the same as q6() except that the environment is different.

<h3> SubQuestion 8 </h3>

As defined in function 'q8()'. It takes no parameters as arguments.

The various parameters for the algorithms are:

    epsilon = 0.5
    steps = 1000
    decay_type = 'linear'
    c = 0.2
    temp = 100

p_exploitation, p_exploration,e_greedy,e_decay,p_softmax, p_ucb are n-dimensional arrays corresponding to the array of tuples of q_est, action taken and rewards of each 50 environments. 
p_exploitation[i][1] is the array of action taken for i-th environment for pure exploitation while qmax is the argmax of final q value (optimal action) corresponding to each algorithms (There are 6 Algorithms)

<h3> SubQuestion 9 </h3>

As defined in function 'q9()'. It takes no parameters as arguments.

The logic and the variables are the same as q8() except that the environment is different.



<h2 name = "question2"> Question 2 (trainQ2.py) </h2>

*All Questions are solved in the function declared as question number. Example, Question 5 is solved in the function declared as q5(), Question 9 as q9(). This is True for Questions 5-10, 12-14. They take no parameter as input

*The policy is ALWAYS LEFT.  

*The seed taken for numpy and the environment is 373. The gym environment is declared as follows:

    env = gym.make('RandomWalk-v0')
    env.seed(373)
    np.random.seed(373)
    env.action_space.seed(373)
    state, _ = env.reset()  

<h3> SubQuestion 1 </h3>

* Declared under 'generateTrajectory(env, policy, maxsteps)'. 
* Returns an empty array if number of steps exceed maxsteps

<h3> SubQuestion 2 </h3>

* Declared under 'decayLearningRate(initialValue, finalValue, maxSteps, decayType)'.
* decayType could be decayType = 'linear' or 'exponential'
* Returns an empty array if number of steps exceed maxsteps
* The plot for linear and exponential decay is plotted under the function 'plot_decayLearningRate()'

<h3> SubQuestion 3 </h3>

* Declared under 'montecarloprediction(env, policy, gamma, alpha, maxSteps, noEpisodes, firstVisit)'
* firstVisit = 1 for EVMC and = 0 for FVMC.
* Returns a tuple of final value of the estimate, the estimate for every episode and the target value for every episode

<h3> SubQuestion 4 </h3>

* Declared under 'TemporalDifferencePrediction(env, policy, gamma, alpha, noEpisodes)'
* Returns a tuple of final value of the estimate, the estimate for every episode and the target value for every episode

<h3> SubQuestion 5 </h3>

* As defined in function 'q5()'. It takes no parameters as arguments.
* The various parameters to the Monte Carlo Prediction function are:
    policy = always left (policy[s] = 0 for all s)
    gamma = 1
    alpha = 0.5
    maxSteps = 100
    No of episodes = 500
    first Visit = 1 (EVMC)

* Plot the estimate that the prediction returns.

<h3> SubQuestion 6 </h3>

* As defined in function 'q6()'. It takes no parameters as arguments.
* The various parameters to the Monte Carlo Prediction function are:
    policy = always left (policy[s] = 0 for all s)
    gamma = 1
    alpha = 0.5
    maxSteps = 100
    No of episodes = 500
    first Visit = 0 (FVMC)

* Plot the estimate that the prediction returns.

<h3> SubQuestion 7 </h3>

* As defined in function 'q7()'. It takes no parameters as arguments.
* The various parameters to the TD Learning function are:
    policy = always left (policy[s] = 0 for all s)
    gamma = 1
    alpha = 0.5
    No of episodes = 500

* Plot the estimate that the prediction returns.

<h3> SubQuestion 8 - 10 </h3>

* Defined as per 'q8()' , 'q9()' and 'q10()'. It takes no parameters as arguments.

* The logic is same as per 'q5()', 'q6()' and 'q7()' [SubQuestions 5 to 7] equivalently except the x-scale is logarithmic. 

<h3> SubQuestion 12 </h3>

* As defined in function 'q12()'. It takes no parameters as arguments.
* The various parameters to the Monte Carlo Prediction function are:
    policy = always left (policy[s] = 0 for all s)
    gamma = 1
    alpha = 0.5
    maxSteps = 100
    No of episodes = 500
    first Visit = 1 (EVMC)

* Plot the target value of state = 3 (state =0,6 are terminal states) that the monte-carlo prediction returns.

<h3> SubQuestion 13 </h3>

* As defined in function 'q12()'. It takes no parameters as arguments.
* The various parameters to the Monte Carlo Prediction function are:
    policy = always left (policy[s] = 0 for all s)
    gamma = 1
    alpha = 0.5
    maxSteps = 100
    No of episodes = 500
    first Visit = 0 (FVMC)

* Plot the target value of state = 3 (state =0,6 are terminal states) that the monte-carlo prediction returns.

<h3> SubQuestion 14 </h3>

* As defined in function 'q12()'. It takes no parameters as arguments.
* The various parameters to the Monte Carlo Prediction function are:
    policy = always left (policy[s] = 0 for all s)
    gamma = 1
    alpha = 0.5
    No of episodes = 500

* Plot the target value of state = 3 (state =0,6 are terminal states) that the TD learning returns.