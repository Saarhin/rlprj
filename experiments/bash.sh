#!/bin/bash

# Define arrays of parameter values
seeds=(0 1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16 17 18 19 20)
algs=("PPO" "DQN")
envs=("CartPole" "Acrobot")
ts=(0.1 0.2 0.4 0.6 0.8 1 1.2)
alphs=(0.01 0.02 0.005)

# Create or overwrite the parameters file
output_file="parameters.txt"
> "$output_file"

# Generate parameter combinations for each seed
for seed in "${seeds[@]}"; do
  for alg in "${algs[@]}"; do
    for env in "${envs[@]}"; do
      for t in "${ts[@]}"; do
        for alph in "${alphs[@]}"; do
          path="./results"
          # Append the parameter combination to the file
          echo "--seed $seed --alg $alg --env $env --t $t --alph $alph --path $path" >> "$output_file"
        done
      done
    done
  done
done