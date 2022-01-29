import torch
import matplotlib.pyplot as plt
from collections import deque
import numpy as np

from dqn_agent import QLAgent
from unityagents import UnityEnvironment



def test_trained_weights(weights_file_path, episodes):
    # Loads trained weights to QLA Agent and 
    

    # Initialize Unity Environment
    env = UnityEnvironment(file_name="Banana_Windows_x86_64\Banana_Windows_x86_64\Banana.exe")
    
    # Initialize new agent with trained weights
    network_config = {  #   network_config (dict) = hidden layer network configuration
    'layers': 2,
    'fc1_units': 64,
    'fc2_units': 64,
    } 
    state_size = 37     # state_size (int)      = 37 [State space with `37` dimensions that contains the agent's velocity and ray-based perception of objects around agent's forward direction]
    action_size = 4     # action_size (int)     = 4  [Discrete 0 (forward), 1 (back), 2 (turn left), 3 (turn right)]
    seed = 1            #   seed  (int)           = 0  []
    agent = QLAgent(state_size, action_size, seed, network_config)
    agent.qnetwork_local.load_state_dict(torch.load(weights_file_path))

    scores = []                        # list containing scores from each episode
    scores_window = deque(maxlen=100)  # last 10 scores

    for i_episode in range(episodes):
        # get the default brain
        brain_name = env.brain_names[0]
        env_info = env.reset(train_mode=False)[brain_name] # reset the environment setting train mode to Falase
        state = env_info.vector_observations[0]            # get the current state
        score = 0                                          # initialize the score
        while True:
            action = agent.act(state).item()
            env_info = env.step(action)[brain_name]
            next_state = env_info.vector_observations[0]
            reward = env_info.rewards[0]
            done = env_info.local_done[0]
            score += reward
            state = next_state
            if done:
                break

        print("\rScore: {}".format(score))

        scores_window.append(score)       # save most recent score
        scores.append(score)              # save most recent score
        print('\rEpisode {}\tAverage Score: {:.2f}'.format(i_episode, np.mean(scores_window)), end="")
        if i_episode % 10 == 0:
            print('\rEpisode {}\tAverage Score: {:.2f}'.format(i_episode, np.mean(scores_window)))

    # plot the scores
    fig = plt.figure()
    
    # define scores axis 
    scores_ax = fig.add_subplot(1,1,1)
    reward_data = scores_ax.plot(range(len(scores)), scores, label = 'Reward')

    average_label = f'Mean {np.mean(scores)}'
    average = scores_ax.plot(range(len(scores)), [np.mean(scores)]*len(scores), label=average_label, linestyle = '--')
    scores_ax.set_xlabel('Episode')
    legend = scores_ax.legend(loc='upper right')
    scores_ax.set_title('Reward Output Using Trained Weights')
    scores_ax.set_ylabel('Reward')
    

    plt.show()
    
if __name__ == "__main__":
    weights_file_path = 'trained_data/optimized_weights_ddqn.pth'
    episodes = 100
    print(f'Starting brain testing of file {weights_file_path} for {episodes} episodes')
    test_trained_weights(weights_file_path, episodes)