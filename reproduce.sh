#!/bin/bash

python play-game.py --agent A3CAgent --env Breakout-v0 --video breakout-a3c-1.mjpeg
python play-game.py --agent MCTSAgent --env Breakout-v0 --video breakout-mcts-1.mjpeg
python play-game.py --agent MCTSA3CAgent --env Breakout-v0 --video breakout-combined-1.mjpeg

python play-game.py --agent A3CAgent --env Pong-v0 --video pong-a3c-1.mjpeg
python play-game.py --agent MCTSAgent --env Pong-v0 --video pong-mcts-1.mjpeg
python play-game.py --agent MCTSA3CAgent --env Pong-v0 --video pong-combined-1.mjpeg
