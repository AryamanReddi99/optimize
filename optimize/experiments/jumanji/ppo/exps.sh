#!/bin/bash
conda activate adam
env_name=Sokoban-v0
python ppo_discrete.py env_name=$env_name optimizer=adam
sleep 10
python ppo_discrete.py env_name=$env_name optimizer=myano myano_gamma=0.0 beta_2=0.92
sleep 10
python ppo_discrete.py env_name=$env_name optimizer=myano myano_gamma=0.5 beta_2=0.92
sleep 10
python ppo_discrete.py env_name=$env_name optimizer=myano myano_gamma=1.0 beta_2=0.92
