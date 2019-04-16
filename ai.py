# AI for Self Driving Car

# Importing the libraries

import numpy as np
import random
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.autograd as autograd
from torch.autograd import Variable

#Creating the architecture of the Neural Network

#inheriting from nn.module
class Network(nn.Module):
    
    #input_size is number of input neurons(5 in the case which describe 3 signals and 2 orientation), 
    #and nb_action is nr of output neurons(3 in this case)
    def __init__(self, input_size, nb_action):
        
        #super inherits the tools from nn.module
        super(Network, self).__init__()
        self.input_size = input_size
        self.nb_action = nb_action
        
        #define connections btw input-hidden and hidden-output(hidden neuron nr:30,can be changed)
        self.fc1 = nn.Linear(input_size, 30)
        self.fc2 = nn.Linear(30, nb_action)
    
        #define forward propogation function
    def forward(self, state):
        #relu as rectifier function
        x = F.relu(self.fc1(state))
        q_values = self.fc2(x)
        return q_values

# Implementing Experience Replay

class ReplayMemory(object):
    
    #capacity is the memory, number of transitions
    def __init__(self, capacity):
        self.capacity = capacity
        self.memory = []
    
    #if capacity is exceeeded, delete 0th memory
    def push(self, event):
        #event is a tuple of 4 elements(last state,new state,last action,the last reward)
        self.memory.append(event)
        if len(self.memory) > self.capacity:
            del self.memory[0]
    
    def sample(self, batch_size):
        #take random samples and reshape
        samples = zip(*random.sample(self.memory, batch_size))
        #store each sample into torch variable
        return map(lambda x: Variable(torch.cat(x, 0)), samples)

# Implementing Deep Q Learning

class Dqn():
    
    def __init__(self, input_size, nb_action, gamma):
        # delay coefficient
        self.gamma = gamma
        # reward window
        self.reward_window = []
        self.model = Network(input_size, nb_action)
        # create 100000 transition memory
        self.memory = ReplayMemory(100000)
        # using adam optimizer, learning rate=0.001, selected low value to learn slowly but surely
        self.optimizer = optim.Adam(self.model.parameters(), lr = 0.001)
        # init last state
        self.last_state = torch.Tensor(input_size).unsqueeze(0)
        
        self.last_action = 0
        self.last_reward = 0
    
    def select_action(self, state):
        # using softmax to explore actions and take temperature as 100 to be more sure of the actions
        probs = F.softmax(self.model(Variable(state, volatile = True))*100) # T=100
        action = probs.multinomial()
        # action is pytorch variable so return real value
        return action.data[0,0]
    
    def learn(self, batch_state, batch_next_state, batch_reward, batch_action):
        # get outputs and next outputs
        outputs = self.model(batch_state).gather(1, batch_action.unsqueeze(1)).squeeze(1)
        next_outputs = self.model(batch_next_state).detach().max(1)[0]
        # get target
        target = self.gamma*next_outputs + batch_reward
        # compute loss of prediction, huber loss is recommended for deep q learning
        td_loss = F.smooth_l1_loss(outputs, target)
        # reinitialize optimizer
        self.optimizer.zero_grad()
        # perform backpropogation
        td_loss.backward(retain_variables = True)
        # update the weights
        self.optimizer.step()
    
    def update(self, reward, new_signal):
        # reach new state
        new_state = torch.Tensor(new_signal).float().unsqueeze(0)
        # update memory since we got new state and old state
        self.memory.push((self.last_state, new_state, torch.LongTensor([int(self.last_action)]), torch.Tensor([self.last_reward])))
        # select action
        action = self.select_action(new_state)
        # if we have more than 100 transitions, time to learn
        if len(self.memory.memory) > 100:
            batch_state, batch_next_state, batch_action, batch_reward = self.memory.sample(100)
            self.learn(batch_state, batch_next_state, batch_reward, batch_action)
        #update last action,state,and reward
        self.last_action = action
        self.last_state = new_state
        self.last_reward = reward
        self.reward_window.append(reward)
        #delete first reward if reward window length bigger than 100
        if len(self.reward_window) > 1000:
            del self.reward_window[0]
        return action
    
    def score(self):
        # score is mean of all rewards
        return sum(self.reward_window)/(len(self.reward_window)+1.)
    
    #functions used for saving and loading the model
    
    def save(self):
        torch.save({'state_dict': self.model.state_dict(),
                    'optimizer' : self.optimizer.state_dict(),
                   }, 'last_brain.pth')
    
    def load(self):
        if os.path.isfile('last_brain.pth'):
            print("=> loading checkpoint... ")
            checkpoint = torch.load('last_brain.pth')
            self.model.load_state_dict(checkpoint['state_dict'])
            self.optimizer.load_state_dict(checkpoint['optimizer'])
            print("done !")
        else:
            print("no checkpoint found...")