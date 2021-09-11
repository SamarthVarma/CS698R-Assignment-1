import gym
import numpy as np
import gym_envs
from numpy import dtype, random
from numpy.core.fromnumeric import argmax
import math
import matplotlib.pyplot as plt
import scipy
from scipy import special

def findmeans(ar):
    means = np.zeros(1000)
    for i in range(1000):
        x_t = 0
        for j in range(50):
            x_t = x_t + ar[j][i]
        means[i] = x_t/50
    return means

def q9():
    np.random.seed(373)
    SD = np.random.uniform(size=50)  
    env = np.ndarray(50, dtype=object)
    q_val = np.ndarray(50,dtype=object)
    for i in range(50):    
        env[i] = gym.make('10ArmGaussian-v0', sd = SD[i])
        env[i].seed(373+i)
        np.random.seed(373+i)
        env[i].action_space.seed(373+i)
        q_val[i] = env[i].reset()
    
    epsilon = 0.5
    steps = 1000
    decay_type = 'linear'
    c = 2
    temp = 100
    p_exploitation = np.ndarray((50,3), dtype=object)
    p_exploration = np.ndarray((50,3), dtype=object)
    e_greedy = np.ndarray((50,3),dtype=object)
    e_decay = np.ndarray((50,3),dtype=object)
    p_softmax = np.ndarray((50,3),dtype=object)
    p_ucb = np.ndarray((50,3),dtype=object)

    for i in range(50):
        #print(q_val)
        p_exploitation[i] = pure_exploitation(env[i],steps)
        p_exploration[i] = pure_exploration(env[i],steps)
        e_greedy[i] = epsilon_greedy(env[i],steps, epsilon)
        e_decay[i] = epsilon_decay(env[i],steps, decay_type)
        p_softmax[i] = softmax(env[i],steps, temp)
        p_ucb[i] = UCB(env[i],steps,c) 

    optimal_percentage = np.zeros((50,6,1000))
    optimal_final = np.zeros((6,1000))
    for i in range(50):
        a = np.zeros((6,1000))
        a[0] = p_exploitation[i][1]
        a[1] = p_exploration[i][1]
        a[2] = e_greedy[i][1]
        a[3] = e_decay[i][1]
        a[4] = p_softmax[i][1]
        a[5] = p_ucb[i][1]  
        qmax = np.argmax(q_val[i])

        for j in range(6):
            e = 0
            for k in range(1000):
                if(a[j][k] == qmax):
                    e = e + 1
                optimal_percentage[i][j][k] = e/(k+1)

        optimal_final = optimal_final + optimal_percentage[i]*100
    
    optimal_final = optimal_final/50

    
    #print(optimal_percentage[1])
    plt.plot(np.arange(1,1001), optimal_final[0], label = "Exploit")
    plt.plot(np.arange(1,1001), optimal_final[1], label = "Explore")
    plt.plot(np.arange(1,1001), optimal_final[2], label = "Epsilon")
    plt.plot(np.arange(1,1001), optimal_final[3], label = "Epsilon-Decay")
    plt.plot(np.arange(1,1001), optimal_final[4], label = "Softmax")
    plt.plot(np.arange(1,1001), optimal_final[5], label = "UCB")
    plt.xlabel("Episodes")
    plt.ylabel("%Optimal")
    plt.legend()
    plt.show()


def q8():
    np.random.seed(373)
    alpha = np.random.uniform(size=50) 
    beta = np.random.uniform(size=50) 
    env = np.ndarray(50, dtype=object)
    for i in range(50):    
        env[i] = gym.make('2ArmBandit-v0', alpha = alpha[i], beta = beta[i])
        env[i].seed(373+i)
        np.random.seed(373+i)
        env[i].action_space.seed(373+i)
    
    epsilon = 0.5
    steps = 1000
    decay_type = 'linear'
    c = 2
    temp = 100
    p_exploitation = np.ndarray((50,3), dtype=object)
    p_exploration = np.ndarray((50,3), dtype=object)
    e_greedy = np.ndarray((50,3),dtype=object)
    e_decay = np.ndarray((50,3),dtype=object)
    p_softmax = np.ndarray((50,3),dtype=object)
    p_ucb = np.ndarray((50,3),dtype=object)

    for i in range(50):
        p_exploitation[i] = pure_exploitation(env[i],steps)
        p_exploration[i] = pure_exploration(env[i],steps)
        e_greedy[i] = epsilon_greedy(env[i],steps, epsilon)
        e_decay[i] = epsilon_decay(env[i],steps, decay_type)
        p_softmax[i] = softmax(env[i],steps, temp)
        p_ucb[i] = UCB(env[i],steps,c) 

    optimal_percentage = np.zeros((50,6,1000))
    optimal_final = np.zeros((6,1000))
    for i in range(50):
        a = np.zeros((6,1000))
        a[0] = p_exploitation[i][1]
        a[1] = p_exploration[i][1]
        a[2] = e_greedy[i][1]
        a[3] = e_decay[i][1]
        a[4] = p_softmax[i][1]
        a[5] = p_ucb[i][1]  
        qmax = np.zeros(6)
        qmax[0] = np.argmax(p_exploitation[i][0][-1])
        qmax[1] = np.argmax(p_exploration[i][0][-1])
        qmax[2] = np.argmax(e_greedy[i][0][-1])
        qmax[3] = np.argmax(e_decay[i][0][-1])
        qmax[4] = np.argmax(p_softmax[i][0][-1])
        qmax[5] = np.argmax(p_ucb[i][0][-1])
        #print(qmax)
        for j in range(6):
            e = 0
            for k in range(1000):
                if(a[j][k] == qmax[j]):
                    e = e + 1
                optimal_percentage[i][j][k] = e/(k+1)

        optimal_final = optimal_final + optimal_percentage[i]*100
    
    optimal_final = optimal_final/50

    
    #print(optimal_percentage[1])
    plt.plot(np.arange(1,1001), optimal_final[0], label = "Exploit")
    plt.plot(np.arange(1,1001), optimal_final[1], label = "Explore")
    plt.plot(np.arange(1,1001), optimal_final[2], label = "Epsilon")
    plt.plot(np.arange(1,1001), optimal_final[3], label = "Epsilon-Decay")
    plt.plot(np.arange(1,1001), optimal_final[4], label = "Softmax")
    plt.plot(np.arange(1,1001), optimal_final[5], label = "UCB")
    plt.xlabel("Episodes")
    plt.ylabel("%Optimal")
    plt.legend()
    plt.show()

""" def q8():
    np.random.seed(373)
    alpha = np.random.uniform(size=50)
    beta = np.random.uniform(size=50) 
    print(alpha,beta)  
    epsilon = 0.5
    steps = 1000
    decay_type = 'linear'
    c = 2
    temp = 100
    p_exploitation = np.ndarray(50, dtype=object)
    p_exploration = np.ndarray(50, dtype=object)
    e_greedy = np.ndarray(50,dtype=object)
    e_decay = np.ndarray(50,dtype=object)
    p_softmax = np.ndarray(50,dtype=object)
    p_ucb = np.ndarray(50,dtype=object)

    for i in range(50):
        env = gym.make('2ArmBandit-v0', alpha = alpha[i], beta = beta[i])
        env.seed(373)
        np.random.seed(373)
        env.action_space.seed(373)
        env.reset()
        p_exploitation[i] = pure_exploitation(env,steps)
        p_exploration[i] = pure_exploration(env,steps)
        e_greedy[i] = epsilon_greedy(env,steps, epsilon)
        e_decay[i] = epsilon_decay(env,steps, decay_type)
        p_softmax[i] = softmax(env,steps, temp)
        p_ucb[i] = UCB(env,steps,c) 

    means = np.zeros((6,1000,2))
    means[0] = np.mean(p_exploitation)
    means[1] = np.mean(p_exploration)
    means[2] = np.mean(e_greedy)
    means[3] = np.mean(e_decay)
    means[4] = np.mean(p_softmax)
    means[5] = np.mean(p_ucb)

    #print(means[1])

    q = np.zeros((6,1000))
    q[0] = np.argmax(means[0],1)
    q[1] = np.argmax(means[1],1)
    q[2] = np.argmax(means[2],1)
    q[3] = np.argmax(means[3],1)
    q[4] = np.argmax(means[4],1)
    q[5] = np.argmax(means[5],1)

    print(q[1])
    v = np.zeros(6)
    for i in range(6):
        v[i] = np.argmax(means[i][-1])

    print(v[1])
    optimal_percentage = np.zeros((6,1000))
    for i in range(steps):
        for j in range(6):
            optimal_percentage[j][i] = optimal_percentage[j][i-1]
            if(q[j][i] == v[j]):
                optimal_percentage[j][i] += 1
    
    print(optimal_percentage[1])
    #plot between optimal_percentage[0]to [6] and 1,1001
    plt.plot(np.arange(1,1001), optimal_percentage[0], label = "1")
    plt.plot(np.arange(1,1001), optimal_percentage[1], label = "2")
    plt.plot(np.arange(1,1001), optimal_percentage[2], label = "3")
    plt.plot(np.arange(1,1001), optimal_percentage[3], label = "4")
    plt.plot(np.arange(1,1001), optimal_percentage[4], label = "5")
    plt.plot(np.arange(1,1001), optimal_percentage[5], label = "6")
    plt.legend()
    plt.show()
 """

def q7():
    np.random.seed(373)
    SD = np.random.uniform(size=50)  
    env = np.ndarray(50, dtype=object)
    q_val = np.ndarray(50,dtype=object)
    for i in range(50):    
        env[i] = gym.make('10ArmGaussian-v0', sd = SD[i])
        env[i].seed(373+i)
        np.random.seed(373+i)
        env[i].action_space.seed(373+i)
        q_val[i] = env[i].reset()

    epsilon = 0.5
    steps = 1000
    decay_type = 'linear'
    c = 0.2
    temp = 100

    q = np.ndarray((6,50), dtype=object)
    action = np.ndarray((6,50), dtype=object)

    tot_regret = np.zeros((6,1000))
    for i in range(50):
        q[0][i], action[0][i] , _ = pure_exploitation(env[i],steps)
        q[1][i], action[1][i] , _ = pure_exploration(env[i],steps)
        q[2][i], action[2][i] , _ = epsilon_greedy(env[i],steps, epsilon)
        q[3][i], action[3][i] , _ = epsilon_decay(env[i],steps, decay_type)
        q[4][i], action[4][i] , _ = softmax(env[i],steps, temp)
        q[5][i], action[5][i] , _ = UCB(env[i],steps,c)
        v = np.zeros(6)
        for j in range(6):
            v[j] = np.max(q[j][i][-1])
        regret = np.zeros((6,1000))
        for j in range(6):
            for k in range(1000):
                regret[j][k] = regret[j][k-1] + abs(v[j] - q[j][i][k][int(action[j][i][k])])

        tot_regret = (tot_regret + regret)

    tot_regret = tot_regret/50

    plt.plot(np.arange(start=1,stop=1001), tot_regret[0], label="Exploit")
    plt.plot(np.arange(start=1,stop=1001), tot_regret[1], label="Explore")
    plt.plot(np.arange(start=1,stop=1001), tot_regret[2], label="Epsilon")
    plt.plot(np.arange(start=1,stop=1001), tot_regret[3], label="Epsilon-Decay")
    plt.plot(np.arange(start=1,stop=1001), tot_regret[4], label="Softmax")
    plt.plot(np.arange(start=1,stop=1001), tot_regret[5], label="UCB")
    plt.xlabel("Episodes")
    plt.ylabel("Regret")
    plt.title("Regret vs Time Step/Episodes")
    plt.legend()
    plt.show()

def q6():
    np.random.seed(373)
    alpha = np.random.uniform(size=50)
    beta = np.random.uniform(size=50)
    epsilon = 0.5
    steps = 1000
    decay_type = 'linear'
    c = 0.2
    temp = 100

    env = np.ndarray(50,dtype=object)
    for i in range(50):    
        env[i] = gym.make('2ArmBandit-v0', alpha = alpha[i], beta = beta[i])
        env[i].seed(373+i)
        np.random.seed(373+i)
        env[i].action_space.seed(373+i)

    q = np.ndarray((6,50), dtype=object)
    action = np.ndarray((6,50), dtype=object)

    tot_regret = np.zeros((6,1000))
    for i in range(50):
        q[0][i], action[0][i] , _ = pure_exploitation(env[i],steps)
        q[1][i], action[1][i] , _ = pure_exploration(env[i],steps)
        q[2][i], action[2][i] , _ = epsilon_greedy(env[i],steps, epsilon)
        q[3][i], action[3][i] , _ = epsilon_decay(env[i],steps, decay_type)
        q[4][i], action[4][i] , _ = softmax(env[i],steps, temp)
        q[5][i], action[5][i] , _ = UCB(env[i],steps,c)
        v = np.zeros(6)
        for j in range(6):
            v[j] = np.max(q[j][i][-1])
        regret = np.zeros((6,1000))
        for j in range(6):
            for k in range(1000):
                regret[j][k] = regret[j][k-1] + abs(v[j] - q[j][i][k][int(action[j][i][k])])

        tot_regret = (tot_regret + regret)

    tot_regret = tot_regret/50

    plt.plot(np.arange(start=1,stop=1001), tot_regret[0], label="Exploit")
    plt.plot(np.arange(start=1,stop=1001), tot_regret[1], label="Explore")
    plt.plot(np.arange(start=1,stop=1001), tot_regret[2], label="Epsilon")
    plt.plot(np.arange(start=1,stop=1001), tot_regret[3], label="Epsilon-Decay")
    plt.plot(np.arange(start=1,stop=1001), tot_regret[4], label="Softmax")
    plt.plot(np.arange(start=1,stop=1001), tot_regret[5], label="UCB")
    plt.xlabel("Time Steps")
    plt.ylabel("Regret")
    plt.title("Regret vs Time Step/Episodes")
    plt.legend()
    plt.show()

def q5():
    np.random.seed(373)
    SD = np.random.uniform(size=50)  
    env = np.ndarray(50, dtype=object)
    q_val = np.ndarray(50,dtype=object)
    for i in range(50):    
        env[i] = gym.make('10ArmGaussian-v0', sd = SD[i])
        env[i].seed(373+i)
        np.random.seed(373+i)
        env[i].action_space.seed(373+i)
        q_val[i] = env[i].reset()

    epsilon = 0.5
    steps = 1000
    decay_type = 'linear'
    c = 0.2
    temp = 100
    p_exploitation = np.ndarray(1000, dtype=object)
    p_exploration = np.ndarray(1000, dtype=object)
    e_greedy = np.ndarray(1000,dtype=object)
    e_decay = np.ndarray(1000,dtype=object)
    p_softmax = np.ndarray(1000,dtype=object)
    p_ucb = np.ndarray(1000,dtype=object)

    for i in range(50):
        _,_,p_exploitation[i] = pure_exploitation(env[i],steps)
        _,_,p_exploration[i] = pure_exploration(env[i],steps)
        _,_,e_greedy[i] = epsilon_greedy(env[i],steps, epsilon)
        _,_,e_decay[i] = epsilon_decay(env[i],steps, decay_type)
        _,_,p_softmax[i] = softmax(env[i],steps, temp)
        _,_,p_ucb[i] = UCB(env[i],steps,c)

    means = np.zeros((6,1000))
    means[0] = findmeans(p_exploitation)
    means[1] = findmeans(p_exploration)  
    means[2] = findmeans(e_greedy)  
    means[3] = findmeans(e_decay)  
    means[4] = findmeans(p_softmax)  
    means[5] = findmeans(p_ucb)     

    plt.plot(np.arange(start=1,stop=1001), means[0], label="Exploit")
    plt.plot(np.arange(start=1,stop=1001), means[1], label="Explore")
    plt.plot(np.arange(start=1,stop=1001), means[2], label="Epsilon")
    plt.plot(np.arange(start=1,stop=1001), means[3], label="Epsilon-Decay")
    plt.plot(np.arange(start=1,stop=1001), means[4], label="Softmax")
    plt.plot(np.arange(start=1,stop=1001), means[5], label="UCB")
    plt.ylabel("Average Reward")
    plt.xlabel("Time Steps")
    plt.legend()
    plt.show()

def q4():
    np.random.seed(373)
    alpha = np.random.uniform(size=50)
    beta = np.random.uniform(size=50)
    epsilon = 0.5
    steps = 1000
    decay_type = 'linear'
    c = 0.2
    temp = 100

    env = np.ndarray(50,dtype=object)
    for i in range(50):    
        env[i] = gym.make('2ArmBandit-v0', alpha = alpha[i], beta = beta[i])
        env[i].seed(373+i)
        np.random.seed(373+i)
        env[i].action_space.seed(373+i)

    p_exploitation = np.ndarray(1000, dtype=object)
    p_exploration = np.ndarray(1000, dtype=object)
    e_greedy = np.ndarray(1000,dtype=object)
    e_decay = np.ndarray(1000,dtype=object)
    p_softmax = np.ndarray(1000,dtype=object)
    p_ucb = np.ndarray(1000,dtype=object)

    for i in range(50):
        _,_,p_exploitation[i] = pure_exploitation(env[i],steps)
        _,_,p_exploration[i] = pure_exploration(env[i],steps)
        _,_,e_greedy[i] = epsilon_greedy(env[i],steps, epsilon)
        _,_,e_decay[i] = epsilon_decay(env[i],steps, decay_type)
        _,_,p_softmax[i] = softmax(env[i],steps, temp)
        _,_,p_ucb[i] = UCB(env[i],steps,c)

    means = np.zeros((6,1000))
    means[0] = findmeans(p_exploitation)
    means[1] = findmeans(p_exploration)  
    means[2] = findmeans(e_greedy)  
    means[3] = findmeans(e_decay)  
    means[4] = findmeans(p_softmax)  
    means[5] = findmeans(p_ucb)     

    plt.plot(np.arange(start=1,stop=1001), means[0], label="Exploit")
    plt.plot(np.arange(start=1,stop=1001), means[1], label="Explore")
    plt.plot(np.arange(start=1,stop=1001), means[2], label="Epsilon")
    plt.plot(np.arange(start=1,stop=1001), means[3], label="Epsilon-Decay")
    plt.plot(np.arange(start=1,stop=1001), means[4], label="Softmax")
    plt.plot(np.arange(start=1,stop=1001), means[5], label="UCB")
    plt.ylabel("Average Reward")
    plt.xlabel("Time Steps")
    plt.legend()
    plt.show()

def UCB(env, ep_count, c):
    x = env.action_space.n
    q = np.zeros(x)
    n = np.zeros(x)
    action = np.zeros(ep_count)
    q_est = np.zeros((ep_count,x))
    e = 0
    r_d = np.zeros(ep_count)
    while(e < ep_count):
        if(e<len(q)): a = e
        else:
            U = [c * np.sqrt(math.log(e)/n)]
            a = argmax(np.add(q,U)) 
        _, r, _, _ = env.step(a)
        n[a] = n[a] + 1
        q[a] = q[a] + (r - q[a])/n[a]
        q_est[e] = q
        action[e] = a
        r_d[e] = r
        e = e + 1
    return q_est, action, r_d

def softmax_eq(q, t):
    z = sum([math.exp(x / t) for x in q])
    probs = [math.exp(x / t) / z for x in q]
    return probs

def softmax(env, ep_count, temp):
    x = env.action_space.n
    q = np.zeros(x)
    n = np.zeros(x)
    action = np.zeros(ep_count)
    q_est = np.zeros((ep_count,x))
    e = 0
    r_d = np.zeros(ep_count)
    t = temp
    while(e < ep_count):
        probs = special.softmax(q/temp)  #from scipy
        a = np.random.choice(env.action_space.n, p = probs)
        _, r, _, _ = env.step(a)
        n[a] = n[a] + 1
        q[a] = q[a] + (r - q[a])/n[a]
        q_est[e] = q
        action[e] = a
        temp = t*math.exp(-math.log(t/0.01)*((e+1)/ep_count))
        r_d[e] = r
        e = e + 1
    return q_est, action, r_d

def epsilon_decay(env, ep_count, decay_type, final_rate = 0.01):
    x = env.action_space.n
    q = np.zeros(x)
    n = np.zeros(x)
    action = np.zeros(ep_count)
    q_est = np.zeros((ep_count,x))
    e = 0
    epsilon = 1
    r_d = np.zeros(ep_count)
    while(e < ep_count):
        if(np.random.rand() > epsilon):
            a = np.argmax(q)
        else:
            a = np.random.randint(0,x)
        _, r, _, _ = env.step(a)
        n[a] = n[a] + 1
        q[a] = q[a] + (r - q[a])/n[a]
        q_est[e] = q
        action[e] = a
        if decay_type == 'linear':
            epsilon = epsilon - (1/ep_count)
        elif decay_type == 'exponential':
            epsilon = math.exp(-math.log(1/0.01)*(e+1)/ep_count)
        r_d[e] = r
        e = e + 1
    return q_est, action, r_d

def epsilon_greedy(env, ep_count, epsilon = 0.5):
    x = env.action_space.n
    q = np.zeros(x)
    n = np.zeros(x)
    action = np.zeros(ep_count)
    q_est = np.zeros((ep_count,x))
    e = 0
    r_d = np.zeros(ep_count)
    while(e < ep_count):
        if(np.random.rand() > epsilon):
            a = np.argmax(q)
        else:
            a = np.random.randint(0,x)
        _, r, _, _ = env.step(a)
        n[a] = n[a] + 1
        q[a] = q[a] + (r - q[a])/n[a]
        q_est[e] = q
        action[e] = a
        r_d[e] = r
        e = e + 1
    return q_est, action, r_d


def pure_exploration(env, ep_count):
    x = env.action_space.n
    q = np.zeros(x)
    n = np.zeros(x)
    action = np.zeros(ep_count)
    q_est = np.zeros((ep_count,x))
    e = 0
    r_d = np.zeros(ep_count)
    while(e < ep_count):
        a = np.random.randint(0,x)
        _, r, _, _ = env.step(a)
        n[a] = n[a] + 1
        q[a] = q[a] + (r - q[a])/n[a]
        q_est[e] = q
        action[e] = a
        r_d[e] = r
        e = e + 1
    return q_est, action, r_d
   

def pure_exploitation(env, ep_count):
    x = env.action_space.n
    q = np.zeros(x)
    n = np.zeros(x)
    action = np.zeros(ep_count)
    q_est = np.zeros((ep_count,x))
    e = 0
    r_d = np.zeros(ep_count)
    while(e < ep_count):
        a = np.argmax(q)
        _, r, _, _ = env.step(a)
        n[a] = n[a] + 1
        q[a] = q[a] + (r - q[a])/n[a]
        q_est[e] = q
        action[e] = a
        r_d[e] = r
        e = e + 1
    return q_est, action, r_d

def doublearmbernoullibandit(hyperparameters, ep_count):
    for i in range(len(hyperparameters['alpha'])):
        env = gym.make('2ArmBandit-v0', alpha = hyperparameters['alpha'][i], beta = hyperparameters['beta'][i])
        env.seed(373)
        np.random.seed(373)
        env.action_space.seed(373)
        env.reset()
        tot_return = 0
        print(f"for alpha = {hyperparameters['alpha'][i]} and beta = {hyperparameters['beta'][i]}")
        for _ in range(ep_count):
            _ , reward, _ , _ = env.step(env.action_space.sample())
            tot_return = tot_return + reward
           # print(reward)

        print(tot_return)
        env.close()

def TenArmGaussian(hyperparameters, ep_count):
    for i in range(len(hyperparameters['sd'])):
        env = gym.make('10ArmGaussian-v0', sd = hyperparameters['sd'][i])
        env.seed(373)
        np.random.seed(373)
        env.action_space.seed(373)
        env.reset()
        tot_return = 0
        print(f"for sd = {hyperparameters['sd'][i]}")
        for _ in range(ep_count):
            _ , reward, _ , _ = env.step(env.action_space.sample())
            tot_return = tot_return + reward
            #print(reward)

        print(tot_return)
        env.close()

if __name__ == '__main__':


    env = '1d0ArmGaussian-v0'

    if env == '2ArmBandit-v0':
        hyperparameters = {
            #'alpha': [0,1,0,1,0.5],
            #'beta': [0,0,1,1,0.5]
            'alpha':[0.3],
            'beta':[0.6],
            'epsilon':0.4
        }
        episode_count = 10000
        env = gym.make('2ArmBandit-v0', alpha = hyperparameters['alpha'], beta = hyperparameters['beta'])
        env.seed(373)
        np.random.seed(373)
        env.action_space.seed(373)
        env.reset()
        print(UCB(hyperparameters, episode_count,0.2))
    
    if env == '10ArmGaussian-v0':
        hyperparameters = {
            'sd': [0,0.2,0.4,0.5,0.75,1,50,100]
        }
        episode_count = 1000
        env = gym.make('10ArmGaussian-v0')
        env.seed(373)
        np.random.seed(373)
        env.action_space.seed(373)
        env.reset()
        print(UCB(env,1000,0.2)[1])


    q7()