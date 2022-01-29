[//]: # (Image References)

[image1]: https://user-images.githubusercontent.com/10624937/42135619-d90f2f28-7d12-11e8-8823-82b970a54d7e.gif "Trained Agent"

### Project 1: Problem Statement

The goal of this project is to train an agent to navigate a Unity environment to maximize rewards. To maximize reward, the agent must collect as many yellow bananas as possible while avoiding blue bananas in Unity’s Banana environment. 
Using a Deep Q Network, based on Pytorch library, and Deep Q Learning algorithm, a final set of optimized weights must generate an optimal solution. The RL agent based of these weights must be able to average a reward of at least 13, over 100 episodes. 

You can view the trained agent in the following video: [click here](https://youtu.be/MOz6D0dSNLA)


#### Environment Notes

![Trained Agent][image1]

A reward of +1 is provided for collecting a yellow banana, and a reward of -1 is provided for collecting a blue banana.  Thus, the goal of your agent is to collect as many yellow bananas as possible while avoiding blue bananas.  

The state space has 37 dimensions and contains the agent's velocity, along with ray-based perception of objects around agent's forward direction.  Given this information, the agent has to learn how to best select actions.  Four discrete actions are available, corresponding to:
- **`0`** - move forward.
- **`1`** - move backward.
- **`2`** - turn left.
- **`3`** - turn right.

The task is episodic, and in order to solve the environment, your agent must get an average score of +13 over 100 consecutive episodes.


### Getting Started

#### Downloading Unity Environment
1. Download the environment from one of the links below.  You need only select the environment that matches your operating system:
    - Linux: [click here](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P1/Banana/Banana_Linux.zip)
    - Mac OSX: [click here](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P1/Banana/Banana.app.zip)
    - Windows (32-bit): [click here](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P1/Banana/Banana_Windows_x86.zip)
    - Windows (64-bit): [click here](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P1/Banana/Banana_Windows_x86_64.zip)
    
    (_For Windows users_) Check out [this link](https://support.microsoft.com/en-us/help/827218/how-to-determine-whether-a-computer-is-running-a-32-bit-version-or-64) if you need help with determining if your computer is running a 32-bit version or 64-bit version of the Windows operating system.

    (_For AWS_) If you'd like to train the agent on AWS (and have not [enabled a virtual screen](https://github.com/Unity-Technologies/ml-agents/blob/master/docs/Training-on-Amazon-Web-Service.md)), then please use [this link](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P1/Banana/Banana_Linux_NoVis.zip) to obtain the environment.


2. Place the file in the root directory, `DRL-QLearning-Udacity-Solution/` folder, and unzip (or decompress) the file. 

#### Download & Install Repository


##### Clone repository
```bash
git clone git@github.com:joselaral/p1-navigation-solution-drl.git
```

##### Install using Miniconda3
``` bash
cd /DRL-QLearning-Udacity-Solution
conda env create -f environment.yml
conda activate drlnd
```

##### Installing using Python 3.6.13
Windows
``` bash
cd /p1-navigation-solution-drl
python3 -m venv venv 
venv/Scripts/activate
python pyp install -r requirements.txt
```

Linux
``` bash
cd /p1-navigation-solution-drl
python3 -m venv venv 
source venv/Scripts/activate
python pip install requirements.txt
```

### Repository Content

model.py
- Deep Q Network model using pytorch library

DQN_Agent
- Agent that interacts with environment and udpates Neural network weights.

main.py
- Runs main environment and environment interatcion
- Pltos average reward scores
- Saves optimized trained weights for Deep Q Network 
- Update path to Banana environment download

test.py
- Runs test on Unity environtment using trained weight

images
- images used in report

trained_data
- optimized_weights_ddqn.pth: Trained weights using

P1 Navigation Report.pde
- Final report
### 

