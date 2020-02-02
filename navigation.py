from unityagents import UnityEnvironment
from agent import Agent
from model import DQN
import numpy as np
import matplotlib.pyplot as plt
import torch

env = UnityEnvironment(file_name="BananaLinux/Banana.x86_64")

# # get the default brain
brain_name = env.brain_names[0]
brain = env.brains[brain_name]

# Initilize agent and train the dqn
agent = Agent(state_size=37, action_size=4, seed=7, duelist=True, prioritize_memory=True)
scores = DQN(env=env, agent=agent, brain_name=brain_name)

#Watch a smart agent

# load the weights from file
agent.qnetwork_local.load_state_dict(torch.load('checkpoint.pth'))

env_info = env.reset(train_mode=False)[brain_name]
state = env_info.vector_observations[0]

for j in range(200):
    action = agent.act(state)
                
    env_info = env.step(action)[brain_name]        # send the action to the environment
    state = env_info.vector_observations[0]   # get the next state
    reward = env_info.rewards[0]                   # get the reward
    done = env_info.local_done[0]
    if done:
        break 
            
env.close()
