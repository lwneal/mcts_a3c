# Baby Monte Carlo Search Tree
from __future__ import print_function
import torch, os, gym, time, glob, argparse
import numpy as np
from scipy.signal import lfilter
from scipy.misc import imresize # preserves single-pixel info _unlike_ img = img[::2,::2]

import torch.nn as nn
from torch.autograd import Variable
import torch.nn.functional as F
from torch.optim import Adam
import imutil

MAX_FRAMES = 1000 * 10


class Node():
    def __init__(self, state, parent):
        self.state = state
        self.children = {}
        self.parent = parent
        self.reward = 


class MCTSAgent():
    def __init__(self, action_space):
        self.action_space = action_space
    
    def act(self, state, env):
        root = Node()
        root_env = env.
        current = root
        for _ in range(100):
            # Selection/Expansion: Find a new leaf node using the Tree Policy
            leaf = self.explore_to_leaf(root, env)

            # Simulation: Perform a "playout" or "rollout" with the Default Policy
            rollout_reward = self.play_to_end(leaf)

            # Backup or Backpropagation: Add to the reward in all parent nodes
            # (NOT to be confused with backpropagation in neural nets)
            while leaf is not None:
                leaf.total_reward += reward
        return 0
    
    def explore_to_leaf(self, root, env):
        node = root
        while True:
            action = self.action_space.sample()  # TODO: Tree Policy
            state = 
            if node.children.get(action) is None:
                node.children[action] = Node(state = 
        return curr


def play(args):
    print("Creating gym with args: {}".format(args))
    env = gym.make(args.env)
    env.unwrapped.frameskip = 10
    state = torch.Tensor(preprocess(env.reset())) # get first state
    agent = MCTSAgent(env.action_space)

    num_frames = 0
    num_games = 0
    cumulative_reward = 0
    while num_frames < MAX_FRAMES:
        # Decide which action to take
        action = agent.act(state, env)

        # Take that action
        state, reward, done, _ = env.step(action)
        cumulative_reward += reward
        if done:
            print("Restarting game after frame {}".format(num_frames))
            num_games += 1
            env.reset()
        state = torch.Tensor(preprocess(state))

        # Record the results
        num_frames += 1
    print("Finished after {} frames with {:.3f} reward/game".format(num_frames, cumulative_reward / num_games))


def preprocess(img):
    return imresize(img[34:194].mean(2), (80,80)).astype(np.float32).reshape(1,80,80) / 255.


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=None)
    parser.add_argument('--env', default='PongNoFrameskip-v4', type=str, help='gym environment')
    parser.add_argument('--lstm_steps', default=20, type=int, help='steps to train LSTM over')
    parser.add_argument('--lr', default=1e-4, type=float, help='learning rate')
    parser.add_argument('--gamma', default=0.99, type=float, help='discount for gamma-discounted rewards')
    parser.add_argument('--tau', default=1.0, type=float, help='discount for generalized advantage estimation')
    parser.add_argument('--horizon', default=0.99, type=float, help='horizon for running averages')
    args = parser.parse_args()

    args.save_dir = '{}/'.format(args.env.lower()) # keep the directory structure simple
    args.num_actions = gym.make(args.env).action_space.n # get the action space of this game

    if not os.path.exists(args.save_dir):
        os.makedirs(args.save_dir)
    print("A3C calling train...")
    play(args)
