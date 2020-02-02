import numpy as np
import random
from collections import namedtuple, deque

from model import QNetwork, DuelingQN
from experience_replay import *
import torch
import torch.nn.functional as F
import torch.optim as optim

BUFFER_SIZE = int(1e5)  # replay buffer size
BATCH_SIZE = 64         # minibatch size
GAMMA = 0.95            # discount factor
TAU = 1e-1              # for soft update of target parameters
LR = 5e-4               # learning rate 
UPDATE_EVERY = 5        # how often to update the network
ALPHA = 1
BETA = 0.4

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


class Agent():
    """Interacts with and learns from the environment."""

    def __init__(self, 
                 state_size,
                 action_size,
                 seed,
                 duelist=False,
                 prioritize_memory=False):
        """Initialize an Agent object.

        Params
        ======
            state_size (int): dimension of each state
            action_size (int): dimension of each action
            seed (int): random seed
            duelist (bool): whether to initialize a dueling Q-Network architecture 
            prioritize_memory (bool): whether to use a prioritized buffer replay
        """
        self.state_size = state_size
        self.action_size = action_size
        self.seed = random.seed(seed)

        # Q-Network
        if duelist:
            self.qnetwork_local = DuelingQN(state_size, action_size, seed).\
                to(device)
            self.qnetwork_target = DuelingQN(state_size, action_size, seed).\
                to(device)
        else:
            self.qnetwork_local = QNetwork(state_size, action_size, seed).\
                to(device)
            self.qnetwork_target = QNetwork(state_size, action_size, seed).\
                to(device)

        self.optimizer = optim.Adam(self.qnetwork_local.parameters(), lr=LR)

        # Replay memory
        if prioritize_memory:
            self.memory = PriorReplayBuffer(action_size, BUFFER_SIZE, BATCH_SIZE, seed, ALPHA, BETA)
            self.prioritize = True
        else:
            self.memory = ReplayBuffer(action_size, BUFFER_SIZE, BATCH_SIZE, seed)
            self.prioritize = False

        # Initialize time step (for updating every UPDATE_EVERY steps)
        self.t_step = 0

    def step(self, state, action, reward, next_state, done):
        # Save experience in replay memory
        self.memory.add(state, action, reward, next_state, done)

        # Learn every UPDATE_EVERY time steps.
        self.t_step = (self.t_step + 1) % UPDATE_EVERY
        if self.t_step == 0:
            # If enough samples are available in memory, get random subset and learn
            if len(self.memory) > BATCH_SIZE:
                experiences = self.memory.sample()
                self.learn(experiences, GAMMA)

    def act(self, state, eps=0.):
        """Returns actions for given state as per current policy.

        Params
        ======
            state (array_like): current state
            eps (float): epsilon, for epsilon-greedy action selection
        """
        state = torch.from_numpy(state).float().unsqueeze(0).to(device)
        self.qnetwork_local.eval()
        with torch.no_grad():
            action_values = self.qnetwork_local(state)
        self.qnetwork_local.train()

        # Epsilon-greedy action selection
        if random.random() > eps:
            return np.argmax(action_values.cpu().data.numpy())
        else:
            return random.choice(np.arange(self.action_size))

    def learn(self, experiences, gamma):         
        if self.prioritize:
            states, actions, rewards, next_states, dones, idxs, weights = experiences
        else:
            states, actions, rewards, next_states, dones = experiences

        # Get expected Q values from local model
        Q_values = self.qnetwork_local(states).gather(1, actions)

        # Get max predicted Q values (for next states) from target model
        next_Q_values = self.qnetwork_target(next_states).detach().max(1)[0].unsqueeze(1)

        # Compute Q targets for current states
        Q_targets = rewards + (gamma * next_Q_values * (1 - dones))
        
        if self.prioritize:
            # Compute loss
            loss = F.mse_loss(Q_values, Q_targets, reduction='none') * weights
            # loss = (Q_values - Q_targets).pow(2) * weights
            new_priorities = loss + 1e-5
            loss = loss.mean()

            # Minimize the loss
            self.optimizer.zero_grad()
            loss.backward()

            # Update experiences priorities
            self.memory.update_priorities(idxs, new_priorities.data.cpu().numpy())
            self.optimizer.step()
        else: 
            # Compute loss
            loss = F.mse_loss(Q_values, Q_targets)
            # Minimize the loss
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

        # ------------------- update target network ------------------- #
        self.soft_update(self.qnetwork_local, self.qnetwork_target, TAU)

    def soft_update(self, local_model, target_model, tau):
        """Soft update model parameters.
        θ_target = τ*θ_local + (1 - τ)*θ_target

        Params
        ======
            local_model (PyTorch model): weights will be copied from
            target_model (PyTorch model): weights will be copied to
            tau (float): interpolation parameter 
        """
        for target_param, local_param in zip(target_model.parameters(), local_model.parameters()):
            target_param.data.copy_(tau*local_param.data + (1.0-tau)*target_param.data)
