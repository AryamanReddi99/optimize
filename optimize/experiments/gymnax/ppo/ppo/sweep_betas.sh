#!/bin/bash

# Activate conda environment
source ~/miniconda3/etc/profile.d/conda.sh
conda activate cheap

# Array of beta_1 values to test
beta_1_values=(0.8 0.9 0.99 1.0)

# Loop through each beta_1 value
for beta_1 in "${beta_1_values[@]}"; do
    echo "Running PPO with beta_1 = $beta_1"
    
    # Run the PPO training script with the current beta_1 value
    python ppo_discrete.py beta_1=$beta_1
    
    echo "Completed run with beta_1 = $beta_1"
    echo "----------------------------------------"
    sleep 120 # sleep for 2 minutes
done

echo "All beta_1 sweeps completed!"
