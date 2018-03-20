# Excerpt from: Baby Advantage Actor-Critic | Sam Greydanus | October 2017 | MIT License
import torch, os, time, glob, argparse, sys
import numpy as np

import torch.nn as nn
from torch.autograd import Variable
import torch.nn.functional as F

SAVE_DIR = 'breakout-v0/'

class NNPolicy(torch.nn.Module): # an actor-critic neural network
    def __init__(self, channels, num_actions):
        super(NNPolicy, self).__init__()
        self.conv1 = nn.Conv2d(channels, 32, 3, stride=2, padding=1)
        self.conv2 = nn.Conv2d(32, 32, 3, stride=2, padding=1)
        self.conv3 = nn.Conv2d(32, 32, 3, stride=2, padding=1)
        self.conv4 = nn.Conv2d(32, 32, 3, stride=2, padding=1)
        self.lstm = nn.LSTMCell(32 * 5 * 5, 256)
        self.critic_linear, self.actor_linear = nn.Linear(256, 1), nn.Linear(256, num_actions)

    def forward(self, inputs, hx, cx):
        x = F.elu(self.conv1(inputs)) ; x = F.elu(self.conv2(x))
        x = F.elu(self.conv3(x)) ; x = F.elu(self.conv4(x))
        hx, cx = self.lstm(x.view(-1, 32 * 5 * 5), (hx, cx))
        return self.critic_linear(hx), self.actor_linear(hx), hx, cx

    def try_load(self, save_dir):
        paths = glob.glob(save_dir + '*.tar') ; step = 0
        if len(paths) > 0:
            ckpts = [int(s.split('.')[-2]) for s in paths]
            ix = np.argmax(ckpts) ; step = ckpts[ix]
            self.load_state_dict(torch.load(paths[ix]))
        print("\tno saved models") if step is 0 else print("\tloaded model: {}".format(paths[ix]))
        return step


class A3CAgent():
    def __init__(self, action_space):
        self.model = NNPolicy(channels=1, num_actions=action_space.n)
        self.model.try_load(SAVE_DIR)
        self.hx = Variable(torch.zeros(1, 256))
        self.cx = Variable(torch.zeros(1, 256))

    def act(self, state, env):
        # Use the neural network!
        state = Variable(state.view(1, 1, 80, 80))
        value, policy, self.hx, self.cx = self.model(state, self.hx, self.cx)
        action = policy.data.numpy().argmax()
        return action
