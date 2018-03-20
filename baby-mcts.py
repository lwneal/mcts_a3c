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
        self.reward = 0
        self.visits = 0


class MCTSAgent():
    def __init__(self, action_space):
        self.action_space = action_space
    
    def act(self, state, env, search_size=100):
        original_state = env.unwrapped.clone_full_state()

        root = Node(state, parent=None)
        for _ in range(search_size):
            # Selection/Expansion: Find a new leaf node using the Tree Policy
            leaf = self.explore_to_leaf(root, env)

            # Simulation: Perform a "playout" or "rollout" with the Default Policy
            playout_reward = self.play_to_end(leaf, env)

            # Backup: Add to the reward in all parent nodes
            while leaf is not None:
                leaf.reward += playout_reward
                leaf.visits += 1
                leaf = leaf.parent

            # Put things back the way they were
            env.unwrapped.restore_full_state(original_state)

        # Select the best action from among the root's children
        best_action = 0
        best_score = -float('inf')
        for action, child in root.children.items():
            score = child.reward / child.visits
            if score > best_score:
                best_score = score
                best_action = action
        print('Best action: {} with expected reward {}'.format(best_action, best_score))
        return best_action
    
    def explore_to_leaf(self, root, env):
        # This is the part where we use the UCB formula
        node = root
        while True:
            action = self.action_space.sample()  # TODO: Tree Policy
            state, reward, done, _ = env.step(action)
            if node.children.get(action) is None:
                # Expansion: Add this node to the tree
                node.children[action] = Node(state, parent=node)
                return node.children[action]
            node = node.children[action]

    def play_to_end(self, leaf, env):
        # This is what you call a 'simulation', a 'playout', or a 'rollout'
        cumulative_reward = 0
        while True:
            action = self.action_space.sample()  # TODO: Default Policy
            state, reward, done, _ = env.step(action)
            cumulative_reward += reward
            # Heuristic: Stop search at any reward
            if reward != 0:
                done = True
            if done:
                break
        return cumulative_reward


def play(args):
    print("Creating gym with args: {}".format(args))
    env = gym.make(args.env)
    env.unwrapped.frameskip = 4
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
        imutil.show(state, video_filename='pure_mcts.mjpeg')
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
    args = parser.parse_args()

    play(args)
