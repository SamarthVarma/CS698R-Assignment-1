import gym
import numpy as np
import math
import matplotlib.pyplot as plt
import gym_envs

def generateTrajectory(env, policy, maxsteps):
    t = []
    state, done = env.reset()
    for i in range(maxsteps):
        action = policy[int(state)]
        next_state, reward, done, _ = env.step(action)
        p = (state,action,reward,next_state)
        t.append(p)
        state = next_state
        if(done): return t
    return []

def decayLearningRate(initialValue, finalValue, maxSteps, decayType):
    decayRate = (initialValue - finalValue)/maxSteps
    alpha = []
    if decayType == 'linear':
        for i in range(maxSteps):
            alpha.append(initialValue - i*decayRate)
        return alpha
    elif decayType == 'exponential':
        for i in range(maxSteps):
            alpha.append(initialValue*math.exp(-math.log(initialValue/finalValue)*(i/maxSteps)))
        return alpha
    
def plot_decayLearningRate():
    linear = decayLearningRate(100, 0.1, 100, 'linear')
    print(linear)
    exponential = decayLearningRate(100, 0.1, 100, 'exponential')
    
    plt.plot(np.arange(1,101),linear, label = "linear")
    plt.plot(np.arange(1,101),exponential, label = "exponential")
    plt.xlabel("No. of Steps")
    plt.ylabel("Alpha value")
    plt.legend()
    plt.show()


def montecarloprediction(env, policy, gamma, alpha, maxSteps, noEpisodes, firstVisit):
    v = np.zeros(7)
    v_r = np.zeros((noEpisodes,7))
    G_t = np.zeros((noEpisodes,7))
    g_s = np.zeros(7)
    alpha_learning = decayLearningRate(0.5, 0.01, 250, 'exponential')
    for e in range(noEpisodes):
        alpha = alpha_learning[min(e,len(alpha_learning)-1)]
        t = generateTrajectory(env, policy, maxSteps)
        visited = np.zeros(7)
        for i, (s,a,r,s_d) in enumerate(t):
            s = int(s)
            if visited[s] and firstVisit:
                continue
            g = 0
            for j in range(i,len(t)):
                g = g + pow(gamma,j-i)*t[j][2]
            g_s[s] = g
            v[s] = v[s] + alpha*(g - v[s])
            visited[s] = 1
        v_r[e] = v
        G_t[e] = g_s
    
    return v, v_r, G_t

def TemporalDifferencePrediction(env, policy, gamma, alpha, noEpisodes):
    v = np.zeros(7)
    v_r = np.zeros((noEpisodes,7))
    G_t = np.zeros((noEpisodes,7))
    g_s = np.zeros(7)
    alpha_learning = decayLearningRate(0.5, 0.01, 250, 'exponential')
    for e in range(noEpisodes):
        alpha = alpha_learning[min(e,len(alpha_learning)-1)]
        s, done = env.reset()
        while not done:
            s = int(s)
            a = policy[s]
            s_d, r, done, _ = env.step(a)
            td_target = r
            if not done: 
                td_target = td_target + gamma*v[int(s_d)]
            
            td_error = td_target - v[s]
            v[s] = v[s] + alpha*td_error
            g_s[s] = td_target
            s = s_d
        v_r[e] = v
        G_t[e] = g_s
    return v, v_r, G_t


def q5():
    env = gym.make('RandomWalk-v0')
    env.seed(373)
    np.random.seed(373)
    env.action_space.seed(373)
    state, _ = env.reset()    
    policy = np.zeros(7)
    v, t_r, _ = montecarloprediction(env,policy,1,0.5,100,500,1)
    plt.plot(np.arange(1,501),t_r[:,1], label = "v(1)")
    plt.plot(np.arange(1,501),t_r[:,2], label = "v(2)")
    plt.plot(np.arange(1,501),t_r[:,3], label = "v(3)")
    plt.plot(np.arange(1,501),t_r[:,4], label = "v(4)")
    plt.plot(np.arange(1,501),t_r[:,5], label = "v(5)")
    plt.plot(np.arange(1,501),np.full(500,v[1]), color='#1f77b4')
    plt.plot(np.arange(1,501),np.full(500,v[2]), color='#ff7f0e')
    plt.plot(np.arange(1,501),np.full(500,v[3]), color='#2ca02c')
    plt.plot(np.arange(1,501),np.full(500,v[4]), color='#d62728')
    plt.plot(np.arange(1,501),np.full(500,v[5]), color='#9467bd')
    plt.xlabel("Episodes")
    plt.ylabel("State- Value Function")
    plt.title("EVMC estimates through time vs True Values")
    plt.legend()
    plt.show()

def q6():
    env = gym.make('RandomWalk-v0')
    env.seed(373)
    np.random.seed(373)
    env.action_space.seed(373)
    state, _ = env.reset()    
    policy = np.zeros(7)
    v, t_r, _ = montecarloprediction(env,policy,1,0.5,100,500,0)
    plt.plot(np.arange(1,501),t_r[:,1], label = "v(1)")
    plt.plot(np.arange(1,501),t_r[:,2], label = "v(2)")
    plt.plot(np.arange(1,501),t_r[:,3], label = "v(3)")
    plt.plot(np.arange(1,501),t_r[:,4], label = "v(4)")
    plt.plot(np.arange(1,501),t_r[:,5], label = "v(5)")
    plt.plot(np.arange(1,501),np.full(500,v[1]), color='#1f77b4')
    plt.plot(np.arange(1,501),np.full(500,v[2]), color='#ff7f0e')
    plt.plot(np.arange(1,501),np.full(500,v[3]), color='#2ca02c')
    plt.plot(np.arange(1,501),np.full(500,v[4]), color='#d62728')
    plt.plot(np.arange(1,501),np.full(500,v[5]), color='#9467bd')
    plt.xlabel("Episodes")
    plt.ylabel("State- Value Function")
    plt.title("FVMC estimates through time vs True Values")
    plt.legend()
    plt.show()

def q7():
    env = gym.make('RandomWalk-v0')
    env.seed(373)
    np.random.seed(373)
    env.action_space.seed(373)
    state, _ = env.reset()    
    policy = np.zeros(7)
    v, t_r, _ = TemporalDifferencePrediction(env,policy,1,0.5,500)
    plt.plot(np.arange(1,501),t_r[:,1], label = "v(1)")
    plt.plot(np.arange(1,501),t_r[:,2], label = "v(2)")
    plt.plot(np.arange(1,501),t_r[:,3], label = "v(3)")
    plt.plot(np.arange(1,501),t_r[:,4], label = "v(4)")
    plt.plot(np.arange(1,501),t_r[:,5], label = "v(5)")
    plt.plot(np.arange(1,501),np.full(500,v[1]), color='#1f77b4')
    plt.plot(np.arange(1,501),np.full(500,v[2]), color='#ff7f0e')
    plt.plot(np.arange(1,501),np.full(500,v[3]), color='#2ca02c')
    plt.plot(np.arange(1,501),np.full(500,v[4]), color='#d62728')
    plt.plot(np.arange(1,501),np.full(500,v[5]), color='#9467bd')
    plt.xlabel("Episodes")
    plt.ylabel("State- Value Function")
    plt.title("TD estimates through time vs True Values")
    plt.legend()
    plt.show()  


def q8():
    env = gym.make('RandomWalk-v0')
    env.seed(373)
    np.random.seed(373)
    env.action_space.seed(373)
    state, _ = env.reset()    
    policy = np.zeros(7)
    v, t_r, _ = montecarloprediction(env,policy,1,0.5,100,500,1)
    plt.plot(np.arange(1,501),t_r[:,1], label = "v(1)")
    plt.plot(np.arange(1,501),t_r[:,2], label = "v(2)")
    plt.plot(np.arange(1,501),t_r[:,3], label = "v(3)")
    plt.plot(np.arange(1,501),t_r[:,4], label = "v(4)")
    plt.plot(np.arange(1,501),t_r[:,5], label = "v(5)")
    plt.plot(np.arange(1,501),np.full(500,v[1]), color='#1f77b4')
    plt.plot(np.arange(1,501),np.full(500,v[2]), color='#ff7f0e')
    plt.plot(np.arange(1,501),np.full(500,v[3]), color='#2ca02c')
    plt.plot(np.arange(1,501),np.full(500,v[4]), color='#d62728')
    plt.plot(np.arange(1,501),np.full(500,v[5]), color='#9467bd')
    plt.xlabel("Episodes")
    plt.ylabel("State- Value Function")
    plt.title("EVMC estimates through time vs True Values (log scale)")
    plt.xscale('log')
    plt.legend()
    plt.show()

def q9():
    env = gym.make('RandomWalk-v0')
    env.seed(373)
    np.random.seed(373)
    env.action_space.seed(373)
    state, _ = env.reset()    
    policy = np.zeros(7)
    v, t_r, _ = montecarloprediction(env,policy,1,0.5,100,500,0)
    plt.plot(np.arange(1,501),t_r[:,1], label = "v(1)")
    plt.plot(np.arange(1,501),t_r[:,2], label = "v(2)")
    plt.plot(np.arange(1,501),t_r[:,3], label = "v(3)")
    plt.plot(np.arange(1,501),t_r[:,4], label = "v(4)")
    plt.plot(np.arange(1,501),t_r[:,5], label = "v(5)")
    plt.plot(np.arange(1,501),np.full(500,v[1]), color='#1f77b4')
    plt.plot(np.arange(1,501),np.full(500,v[2]), color='#ff7f0e')
    plt.plot(np.arange(1,501),np.full(500,v[3]), color='#2ca02c')
    plt.plot(np.arange(1,501),np.full(500,v[4]), color='#d62728')
    plt.plot(np.arange(1,501),np.full(500,v[5]), color='#9467bd')
    plt.xlabel("Episodes")
    plt.ylabel("State- Value Function")
    plt.title("FVMC estimates through time vs True Values (log scale)")
    plt.xscale('log')
    plt.legend()
    plt.show()

def q10():
    env = gym.make('RandomWalk-v0')
    env.seed(373)
    np.random.seed(373)
    env.action_space.seed(373)
    state, _ = env.reset()    
    policy = np.zeros(7)
    v, t_r, _ = TemporalDifferencePrediction(env,policy,1,0.5,500)
    plt.plot(np.arange(1,501),t_r[:,1], label = "v(1)")
    plt.plot(np.arange(1,501),t_r[:,2], label = "v(2)")
    plt.plot(np.arange(1,501),t_r[:,3], label = "v(3)")
    plt.plot(np.arange(1,501),t_r[:,4], label = "v(4)")
    plt.plot(np.arange(1,501),t_r[:,5], label = "v(5)")
    plt.plot(np.arange(1,501),np.full(500,v[1]), color='#1f77b4')
    plt.plot(np.arange(1,501),np.full(500,v[2]), color='#ff7f0e')
    plt.plot(np.arange(1,501),np.full(500,v[3]), color='#2ca02c')
    plt.plot(np.arange(1,501),np.full(500,v[4]), color='#d62728')
    plt.plot(np.arange(1,501),np.full(500,v[5]), color='#9467bd')
    plt.xlabel("Episodes")
    plt.ylabel("State- Value Function")
    plt.title("TD estimates through time vs True Values (log scale)")
    plt.xscale('log')
    plt.legend()
    plt.show()    

def q12():
    env = gym.make('RandomWalk-v0')
    env.seed(373)
    np.random.seed(373)
    env.action_space.seed(373)
    state, _ = env.reset()    
    policy = np.zeros(7)
    v, v_r, G_t = montecarloprediction(env,policy,1,0.5,100,500,1) 
    print(G_t[:,3])
    plt.plot(np.arange(1,501),np.full(500,v[3]), label = "v(3)", color="green" )
    plt.scatter(np.arange(1,501),G_t[:,3], s=1)
    plt.xlabel("Episodes")
    plt.ylabel("Target Value")
    plt.title("EVMC target sequence")
    plt.legend()
    plt.show()

def q13():
    env = gym.make('RandomWalk-v0')
    env.seed(373)
    np.random.seed(373)
    env.action_space.seed(373)
    state, _ = env.reset()    
    policy = np.zeros(7)
    v, v_r, G_t = montecarloprediction(env,policy,1,0.5,100,500,0) 
    print(G_t[:,3])
    plt.plot(np.arange(1,501),np.full(500,v[3]), label = "v(3)", color="green" )
    plt.scatter(np.arange(1,501),G_t[:,3], s=1)
    plt.xlabel("Episodes")
    plt.ylabel("Target Value")
    plt.title("FVMC target sequence")
    plt.legend()
    plt.show()

def q14():
    env = gym.make('RandomWalk-v0')
    env.seed(373)
    np.random.seed(373)
    env.action_space.seed(373)
    state, _ = env.reset()    
    policy = np.zeros(7)
    v, v_r, G_t = TemporalDifferencePrediction(env,policy,1,0.5,500) 
    print(G_t[:,3])
    plt.plot(np.arange(1,501),np.full(500,v[3]), label = "v(3)", color="green" )
    plt.scatter(np.arange(1,501),G_t[:,3], s=1)
    plt.xlabel("Episodes")
    plt.ylabel("Target Value")
    plt.title("TD target sequence")
    plt.legend()
    plt.show()



if __name__ == '__main__':
    q10()