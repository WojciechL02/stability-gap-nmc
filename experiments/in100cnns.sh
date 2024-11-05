#!/bin/bash

set -e

eval "$(conda shell.bash hook)"
conda activate FACIL

num_tasks=10
nc_first_task=10
stop_at_task=0
dataset=imagenet_subset_kaggle
tag=other_cnns  # experiment name

num_epochs=100
lr=0.1
bsz=128
head_init=zeros

for seed in 0 1 2; do
  ./experiments/ft.sh 0 ${seed} ${tag} ${dataset} ${num_tasks} ${nc_first_task} resnet50 ${num_epochs} 0 0 0 ${lr} ${head_init} ${stop_at_task} ${exemplars} ${bsz} linear &
done
wait

for seed in 0 1 2; do
  ./experiments/ft.sh 0 ${seed} ${tag} ${dataset} ${num_tasks} ${nc_first_task} resnet50 ${num_epochs} 0 0 0 ${lr} ${head_init} ${stop_at_task} ${exemplars} ${bsz} nmc &
done
wait

for seed in 0 1 2; do
  ./experiments/ft.sh 0 ${seed} ${tag} ${dataset} ${num_tasks} ${nc_first_task} vgg11 ${num_epochs} 0 0 0 ${lr} ${head_init} ${stop_at_task} ${exemplars} ${bsz} linear &
done
wait

for seed in 0 1 2; do
  ./experiments/ft.sh 0 ${seed} ${tag} ${dataset} ${num_tasks} ${nc_first_task} vgg11 ${num_epochs} 0 0 0 ${lr} ${head_init} ${stop_at_task} ${exemplars} ${bsz} nmc &
done
wait

for seed in 0 1 2; do
  ./experiments/ft.sh 0 ${seed} ${tag} ${dataset} ${num_tasks} ${nc_first_task} mobilenet_v2 ${num_epochs} 0 0 0 ${lr} ${head_init} ${stop_at_task} ${exemplars} ${bsz} linear &
done
wait

for seed in 0 1 2; do
  ./experiments/ft.sh 0 ${seed} ${tag} ${dataset} ${num_tasks} ${nc_first_task} mobilenet_v2 ${num_epochs} 0 0 0 ${lr} ${head_init} ${stop_at_task} ${exemplars} ${bsz} nmc &
done
wait

for seed in 0 1 2; do
  ./experiments/ft.sh 0 ${seed} ${tag} ${dataset} ${num_tasks} ${nc_first_task} efficientnet_b4 ${num_epochs} 0 0 0 ${lr} ${head_init} ${stop_at_task} ${exemplars} ${bsz} linear &
done
wait

for seed in 0 1 2; do
  ./experiments/ft.sh 0 ${seed} ${tag} ${dataset} ${num_tasks} ${nc_first_task} efficientnet_b4 ${num_epochs} 0 0 0 ${lr} ${head_init} ${stop_at_task} ${exemplars} ${bsz} nmc &
done
wait
