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

from mcts_agent import MCTSAgent
from a3c_agent import A3CAgent
from mctsa3c_agent import MCTSA3CAgent

MAX_FRAMES = 200


def play(args, AgentType):
    print("Creating gym with args: {}".format(args))
    env = gym.make(args.env)
    env.unwrapped.frameskip = 4
    env._max_episode_steps = None
    state = torch.Tensor(preprocess(env.reset())) # get first state
    agent = AgentType(env.action_space)
    start_time = time.time()

    num_frames = 0
    num_games = 0
    cumulative_reward = 0
    while num_frames < MAX_FRAMES:
        # Decide which action to take
        action = agent.act(state, env)

        # Take that action
        state, reward, done, _ = env.step(action)
        cumulative_reward += reward
        imutil.show(state, video_filename=args.video, display=(num_frames%10 == 0))
        state = torch.Tensor(preprocess(state))
        #imutil.show(state, save=False)
        if done:
            print("Restarting game after frame {}".format(num_frames))
            num_games += 1
            env.reset()
        num_frames += 1
    num_games += 1
    print("Finished after {} frames with {:.3f} reward/game".format(num_frames, cumulative_reward / num_games))
    print("Speed: {:.3f} frames per second".format(num_frames / (time.time() - start_time)))


def preprocess(img):
    return imresize(img[34:194].mean(2), (80,80)).astype(np.float32).reshape(1,80,80) / 255.


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=None)
    parser.add_argument('--env', default='PongNoFrameskip-v4', type=str, help='OpenAI Gym environment')
    parser.add_argument('--agent', default='MCTSAgent', type=str, help='Agent to Run')
    parser.add_argument('--video', default='video.mjpeg', type=str, help='Video output filename (MJPEG)')
    args = parser.parse_args()
    AgentType = globals()[args.agent]
    play(args, AgentType)
