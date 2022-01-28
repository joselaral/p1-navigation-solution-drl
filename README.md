[//]: # (Image References)

[image1]: https://user-images.githubusercontent.com/10624937/42135619-d90f2f28-7d12-11e8-8823-82b970a54d7e.gif "Trained Agent"


### Introduction

DRL-QLearning-Udacity-Solution is the a repository that solves the first project in the Udacity's Deep Reinfocement Learning Course. Using Q Learning 


### Problem Statement

#### Project 1: Navigation



#### Environment Notes

![Trained Agent][Benchmark]

A reward of +1 is provided for collecting a yellow banana, and a reward of -1 is provided for collecting a blue banana.  Thus, the goal of your agent is to collect as many yellow bananas as possible while avoiding blue bananas.  

The state space has 37 dimensions and contains the agent's velocity, along with ray-based perception of objects around agent's forward direction.  Given this information, the agent has to learn how to best select actions.  Four discrete actions are available, corresponding to:
- **`0`** - move forward.
- **`1`** - move backward.
- **`2`** - turn left.
- **`3`** - turn right.

The task is episodic, and in order to solve the environment, your agent must get an average score of +13 over 100 consecutive episodes.


#### Implementation Notes




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
git clone XXXXXX
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
cd /DRL-QLearning-Udacity-Solution
python3 -m venv venv 
venv/Scripts/activate
python pyp install -r requirements.txt
```

Linux
``` bash
cd /DRL-QLearning-Udacity-Solution
python3 -m venv venv 
source venv/Scripts/activate
python pip install requirements.txt
```

### Repository Content



model.pyp
Neural Network using pytoch library

DQN_Agent
Agent that interacts with environment and udpates Neural network weights.

main.py
- Runs main environment and environment interatcion
- Stores average reward scores
- Plots results


### Training Q Learning Agent





### 