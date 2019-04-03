#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Wed Apr  3 14:26:09 2019

@author: pauljoegeroge
"""

import numpy as np
import random 
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.autograd as autograd
from torch.autograd import Variable

class Network(nn.Module):
    
    def __init__(self, input_size, nb_action):  #no of input and output neurons. input is 5, nb_action
        super(Network, self).__init__() #helps to use the tools within the function
        self.input_size = input_size
        self.nb_action = nb_action
        #full connection between input layer and hidden layer and then between hiddhen layer and output layer
        self.fc1 = nn.Linear(input_size, 30) #full connection between input and hidden layer (5 input neurons and 30 hidden neurons)
        self.fc2 = nn.Linear(30, nb_action)
        
    #activate neurons, return Q values
    def forward(self, state):
        x = 