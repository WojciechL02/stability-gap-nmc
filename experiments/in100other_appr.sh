#!/bin/bash

set -e

eval "$(conda shell.bash hook)"
conda activate FACIL

num_tasks=10
nc_first_task=10
stop_at_task=0
dataset=imagenet_subset_kaggle
network=resnet18
tag=other_appr  # experiment name

num_epochs=100
lr=0.1
lamb=10000
head_init=zeros

for seed in 0 1 2; do
  ./experiments/lwf.sh 0 ${seed} ${tag} ${dataset} ${num_tasks} ${nc_first_task} ${network} ${num_epochs} 1 0 0 0.0 ${lr} ${head_init} ${stop_at_task} linear &
done
wait

for seed in 0 1 2; do
  ./experiments/lwf.sh 0 ${seed} ${tag} ${dataset} ${num_tasks} ${nc_first_task} ${network} ${num_epochs} 1 0 0 0.0 ${lr} ${head_init} ${stop_at_task} nmc &
done
wait

for seed in 0 1 2; do
  ./experiments/ewc.sh 0 ${seed} ${tag} ${dataset} ${num_tasks} ${nc_first_task} ${network} ${num_epochs} 0 0 0.0 ${lr} 10000 ${head_init} ${stop_at_task} linear &
done
wait

for seed in 0 1 2; do
  ./experiments/ewc.sh 0 ${seed} ${tag} ${dataset} ${num_tasks} ${nc_first_task} ${network} ${num_epochs} 0 0 0.0 ${lr} 10000 ${head_init} ${stop_at_task} nmc &
done
wait

for seed in 0 1 2; do
  ./experiments/ssil.sh 0 ${seed} ${tag} ${dataset} ${num_tasks} ${nc_first_task} ${network} ${num_epochs} 1 0 0 0.0 0.05 ${head_init} ${stop_at_task} linear &
done
wait

for seed in 0 1 2; do
  ./experiments/ssil.sh 0 ${seed} ${tag} ${dataset} ${num_tasks} ${nc_first_task} ${network} ${num_epochs} 1 0 0 0.0 0.05 ${head_init} ${stop_at_task} nmc &
done
wait
