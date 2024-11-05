#!/bin/bash

set -e

eval "$(conda shell.bash hook)"
conda activate FACIL

num_tasks=10
nc_first_task=10
stop_at_task=0
network=resnet18
tag=slca  # experiment name

num_epochs=100
bsz=128
lr=0.1
exemplars=10
head_init=zeros

# FINETUNING
for seed in 0 1 2; do
    ./experiments/ft_pretr.sh 0 ${seed} ${tag} aircrafts ${num_tasks} ${nc_first_task} ${network} ${num_epochs} 0 0 0 ${lr} ${head_init} ${stop_at_task} ${exemplars} ${bsz} linear &
done
wait

for seed in 0 1 2; do
    ./experiments/ft_pretr.sh 0 ${seed} ${tag} birds ${num_tasks} ${nc_first_task} ${network} ${num_epochs} 0 0 0 ${lr} ${head_init} ${stop_at_task} ${exemplars} ${bsz} linear &
done
wait

# FINETUNING + NMC
for seed in 0 1 2; do
    ./experiments/ft_pretr.sh 0 ${seed} ${tag} aircrafts ${num_tasks} ${nc_first_task} ${network} ${num_epochs} 0 0 0 ${lr} ${head_init} ${stop_at_task} ${exemplars} ${bsz} nmc &
done
wait

for seed in 0 1 2; do
    ./experiments/ft_pretr.sh 0 ${seed} ${tag} birds ${num_tasks} ${nc_first_task} ${network} ${num_epochs} 0 0 0 ${lr} ${head_init} ${stop_at_task} ${exemplars} ${bsz} nmc &
done
wait


