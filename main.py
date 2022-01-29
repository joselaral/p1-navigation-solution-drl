import torch
import matplotlib.pyplot as plt
import numpy as np

from dqn_agent import QLAgent
from collections import deque
from unityagents import UnityEnvironment

    
# Initilize Q-Learning Agent with the following inputs:   
# TODO rewrite network config to programatically build network (QNN in model.py) based on these settings
network_config = {  #   network_config (dict) = hidden layer network configuration
    'layers': 2,
    'fc1_units': 64,
    'fc2_units': 64,
} 
state_size = 37     # state_size (int)      = 37 [State space with `37` dimensions that contains the agent's velocity and ray-based perception of objects around agent's forward direction]
action_size = 4     # action_size (int)     = 4  [Discrete 0 (forward), 1 (back), 2 (turn left), 3 (turn right)]
seed = 0            #   seed  (int)           = 0  []
agent = QLAgent(state_size, action_size, seed, network_config)

# Initialize Unity Environment
env = UnityEnvironment(file_name="Banana_Windows_x86_64\Banana_Windows_x86_64\Banana.exe")

# get the default brain
brain_name = env.brain_names[0]
brain = env.brains[brain_name]

def dqn(n_episodes=1800, max_t=1000, eps_start=1.0, eps_end=0.01, eps_decay=0.995):
    """Deep Q-Learning.
    
    Params
    ======
        n_episodes (int): maximum number of training episodes
        max_t (int): maximum number of timesteps per episode
        eps_start (float): starting value of epsilon, for epsilon-greedy action selection
        eps_end (float): minimum value of epsilon
        eps_decay (float): multiplicative factor (per episode) for decreasing epsilon
    """
    scores = []                        # list containing scores from each episode
    avg_scores = []                    # List of average score per 100 episodes
    scores_window = deque(maxlen=100)  # last 100 scores
    eps = eps_start                    # initialize epsilon
    for i_episode in range(1, n_episodes+1):
        env_info = env.reset(train_mode=True)[brain_name] # reset the environment
        state = env_info.vector_observations[0]  
        score = 0
        for t in range(max_t):
            # Update variables used for next step
            action = agent.act(state, eps)
            env_info = env.step(action)[brain_name]        # send the action to the environment
            next_state = env_info.vector_observations[0]   # get the next state
            reward = env_info.rewards[0]                   # get the reward
            done = env_info.local_done[0] 

            # Step through Q Learning agent
            agent.step(state, action, reward, next_state, done)
            state = next_state

            # Update reward
            score += reward
            if done:
                break 

        
        scores_window.append(score)       # save most recent score
        scores.append(score)              # save most recent score
        eps = max(eps_end, eps_decay*eps) # decrease epsilon
        print('\rEpisode {}\tAverage Score: {:.2f}'.format(i_episode, np.mean(scores_window)), end="")
        if i_episode % 100 == 0:
            print('\rEpisode {}\tAverage Score: {:.2f}'.format(i_episode, np.mean(scores_window)))
            avg_scores.append(np.mean(scores_window))

    torch.save(agent.qnetwork_local.state_dict(), 'optimized_weights.pth')

    return scores, avg_scores


def main():
    # Initialize interation between QLearning agent and environment 
    scores, avg_scores = dqn()

    # plot the scores
    fig = plt.figure()
    
    # define scores axis 
    scores_ax = fig.add_subplot(1,2,1)
    scores_ax.plot(range(len(scores)), scores)
    scores_ax.set_xlabel('Episode')
    scores_ax.set_title('Episodic scores')
    scores_ax.set_ylabel('Reward')
    

    avg_scores_ax = fig.add_subplot(1,2,2)
    avg_scores_x = [i*100 for i in range(len(avg_scores))]
    avg_scores_ax.plot(avg_scores_x, avg_scores, '-o')
    avg_scores_ax.set_xlabel('Episode')
    avg_scores_ax.set_title('Average Scores Every 100 Episodes')

    plt.show()

if __name__ == "__main__":
    print(f'\nStarting training')
    main()