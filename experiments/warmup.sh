#!/bin/bash

set -e

eval "$(conda shell.bash hook)"
conda activate FACIL

stop_at_task=0
dataset=cifar100_icarl
network=resnet18
tag=warmup  # experiment name

num_epochs=100
lr=0.1
bsz=128
wu_nepochs=20
wu_lr=0.1
wu_wd=0.0
head_init=zeros

for seed in 0 1 2; do
  ./experiments/ft.sh 0 ${seed} ${tag} ${dataset} 10 10 ${network} ${num_epochs} ${wu_nepochs} ${wu_lr} ${wu_wd} ${lr} ${head_init} ${stop_at_task} ${exemplars} ${bsz} linear &
done
wait

for seed in 0 1 2; do
  ./experiments/ft.sh 0 ${seed} ${tag} ${dataset} 5 20 ${network} ${num_epochs} ${wu_nepochs} ${wu_lr} ${wu_wd} ${lr} ${head_init} ${stop_at_task} ${exemplars} ${bsz} linear &
done
wait
