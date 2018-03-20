import numpy as np
import torch, os, time, glob, argparse, sys
import numpy as np
from scipy.misc import imresize

import torch.nn as nn
from torch.autograd import Variable
import torch.nn.functional as F

SAVE_DIR = 'breakout-v0/'

class Node():
    def __init__(self, state, parent):
        self.state = state
        self.children = {}
        self.parent = parent
        self.reward = 0
        self.visits = 0

def preprocess(img):
    return imresize(img[34:194].mean(2), (80,80)).astype(np.float32).reshape(1,80,80) / 255.


class MCTSA3CAgent():
    def __init__(self, action_space):
        self.action_space = action_space
        # An internal network used as the default policy during rollouts
        self.agent = A3CAgent(action_space)
    
    def act(self, state, env, search_size=100):
        original_state = env.unwrapped.clone_full_state()
        self.agent.save_state()

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
            self.agent.restore_state()

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
            action = self.tree_policy(node)
            state, reward, done, _ = env.step(action)
            if node.children.get(action) is None:
                # Expansion: Add this node to the tree
                node.children[action] = Node(state, parent=node)
                return node.children[action]
            node = node.children[action]

    def tree_policy(self, node):
        # First try any unexplored action
        for a in range(self.action_space.n):
            if a not in node.children:
                return a
        # Then try the UCB-optimal action
        def ucb(node):
            return node.reward / node.visits + np.sqrt(np.log(node.parent.visits) / node.visits)
        return max(node.children, key=lambda a: ucb(node.children[a]))

    def play_to_end(self, leaf, env, depth_limit=0):
        # This is what you call a 'simulation', a 'playout', or a 'rollout'
        cumulative_reward = 0
        depth = 0
        while True:
            input_state = torch.Tensor(preprocess(leaf.state))
            action, val = self.agent.act(input_state, env)
            state, reward, done, _ = env.step(action)
            cumulative_reward += reward
            # Heuristic: Stop search at any reward
            if reward != 0:
                done = True
            depth += 1
            if depth > depth_limit:
                done = True
            if done:
                cumulative_reward += val
                break
        return cumulative_reward


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
        val = value.data.numpy()[0][0]
        return action, val
    
    def save_state(self):
        self.hx_saved = self.hx.clone()
        self.cx_saved = self.cx.clone()

    def restore_state(self):
        self.hx = self.hx_saved
        self.cx = self.cx_saved
