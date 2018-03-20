#!/bin/bash
pip install gym
pip install gym[atari]
echo "running baby-a3c"
python baby-a3c.py
echo "finished baby-a3c"
