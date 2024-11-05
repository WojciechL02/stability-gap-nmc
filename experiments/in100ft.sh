#!/bin/bash

set -e

eval "$(conda shell.bash hook)"
conda activate FACIL

stop_at_task=0
dataset=imagenet_subset_kaggle
network=resnet18
tag=main  # experiment name

num_epochs=100
lr=0.1
bsz=128
head_init=zeros

for seed in 0 1 2; do
  ./experiments/ft.sh 0 ${seed} ${tag} ${dataset} 10 10 ${network} ${num_epochs} 0 0 0.0 ${lr} ${head_init} ${stop_at_task} ${exemplars} ${bsz} linear &
done
wait

for seed in 0 1 2; do
  ./experiments/ft.sh 0 ${seed} ${tag} ${dataset} 10 10 ${network} ${num_epochs} 0 0 0.0 ${lr} ${head_init} ${stop_at_task} ${exemplars} ${bsz} nmc &
done
wait

for seed in 0 1 2; do
  ./experiments/ft_oracle.sh 0 ${seed} ${tag} ${dataset} 10 10 ${network} ${num_epochs} 0 0 0.0 ${lr} ${head_init} ${stop_at_task} ${exemplars} ${bsz} nmc &
done
wait

for seed in 0 1 2; do
  ./experiments/ft.sh 0 ${seed} ${tag} ${dataset} 5 20 ${network} ${num_epochs} 0 0 0.0 ${lr} ${head_init} ${stop_at_task} ${exemplars} ${bsz} linear &
done
wait

for seed in 0 1 2; do
  ./experiments/ft.sh 0 ${seed} ${tag} ${dataset} 5 20 ${network} ${num_epochs} 0 0 0.0 ${lr} ${head_init} ${stop_at_task} ${exemplars} ${bsz} nmc &
done
wait

for seed in 0 1 2; do
  ./experiments/ft_oracle.sh 0 ${seed} ${tag} ${dataset} 5 20 ${network} ${num_epochs} 0 0 0.0 ${lr} ${head_init} ${stop_at_task} ${exemplars} ${bsz} nmc &
done
wait
