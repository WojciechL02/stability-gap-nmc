#!/bin/bash

set -e

eval "$(conda shell.bash hook)"
conda activate FACIL

stop_at_task=0
dataset=cifar100_icarl
network=resnet18
tag=memory  # experiment name

num_epochs=100
lr=0.1
bsz=128
head_init=zeros

for seed in 0 1 2; do
  ./experiments/ft.sh 0 ${seed} ${tag} ${dataset} 5 20 ${network} ${num_epochs} 0 0 0.0 ${lr} ${head_init} ${stop_at_task} 100 ${bsz} linear &
done
wait

for seed in 0 1 2; do
  ./experiments/ft.sh 0 ${seed} ${tag} ${dataset} 5 20 ${network} ${num_epochs} 0 0 0.0 ${lr} ${head_init} ${stop_at_task} 100 ${bsz} nmc &
done
wait

for seed in 0 1 2; do
  ./experiments/ft.sh 0 ${seed} ${tag} ${dataset} 5 20 ${network} ${num_epochs} 0 0 0.0 ${lr} ${head_init} ${stop_at_task} 500 ${bsz} linear &
done
wait

for seed in 0 1 2; do
  ./experiments/ft.sh 0 ${seed} ${tag} ${dataset} 5 20 ${network} ${num_epochs} 0 0 0.0 ${lr} ${head_init} ${stop_at_task} 500 ${bsz} nmc &
done
wait

for seed in 0 1 2; do
  ./experiments/ft.sh 0 ${seed} ${tag} ${dataset} 5 20 ${network} ${num_epochs} 0 0 0.0 ${lr} ${head_init} ${stop_at_task} 1000 ${bsz} linear &
done
wait

for seed in 0 1 2; do
  ./experiments/ft.sh 0 ${seed} ${tag} ${dataset} 5 20 ${network} ${num_epochs} 0 0 0.0 ${lr} ${head_init} ${stop_at_task} 1000 ${bsz} nmc &
done
wait

for seed in 0 1 2; do
  ./experiments/ft.sh 0 ${seed} ${tag} ${dataset} 5 20 ${network} ${num_epochs} 0 0 0.0 ${lr} ${head_init} ${stop_at_task} 5000 ${bsz} linear &
done
wait

for seed in 0 1 2; do
  ./experiments/ft.sh 0 ${seed} ${tag} ${dataset} 5 20 ${network} ${num_epochs} 0 0 0.0 ${lr} ${head_init} ${stop_at_task} 5000 ${bsz} nmc &
done
wait
