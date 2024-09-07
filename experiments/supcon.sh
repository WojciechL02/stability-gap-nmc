#!/bin/bash

set -e

gpu=$1
seed=$2
tag=$3
dataset=$4
num_tasks=$5
nc_first_task=$6
network=$7
num_epochs=$8
lr=${9:-0.1}
head_init=${10}
stop_at_task=${11:-0}
update_prototypes=${12:-0}
exemplars=${13:-20}
temperature=${14:-0.1}
batch_size=${15:-128}


if [ ${update_prototypes} -gt 0 ]; then
    exp_name="t${num_tasks}s${nc_first_task}_hz_m:${exemplars}_up:${update_prototypes}"
    result_path="results/${tag}/supcon_hz_${seed}"
    python3 src/main_incremental.py \
    --exp-name ${exp_name} \
    --gpu ${gpu} \
    --datasets ${dataset} \
    --num-tasks ${num_tasks} \
    --nc-first-task ${nc_first_task} \
    --network ${network} \
    --use-test-as-val \
    --lr ${lr} \
    --nepochs ${num_epochs} \
    --batch-size ${batch_size} \
    --seed ${seed} \
    --log disk wandb \
    --results-path ${result_path} \
    --tags ${tag} \
    --scheduler-milestones \
    --stop-at-task ${stop_at_task} \
    --approach supcon \
    --temperature ${temperature} \
    --num-exemplars ${exemplars} \
    --head-init-mode ${head_init} \
    --update_prototypes
else
    exp_name="t${num_tasks}s${nc_first_task}_hz_m:${exemplars}_up:${update_prototypes}"
    result_path="results/${tag}/supcon_hz_${seed}"
    python3 src/main_incremental.py \
    --exp-name ${exp_name} \
    --gpu ${gpu} \
    --datasets ${dataset} \
    --num-tasks ${num_tasks} \
    --nc-first-task ${nc_first_task} \
    --network ${network} \
    --use-test-as-val \
    --lr ${lr} \
    --nepochs ${num_epochs} \
    --batch-size ${batch_size} \
    --seed ${seed} \
    --log disk wandb \
    --results-path ${result_path} \
    --tags ${tag} \
    --scheduler-milestones \
    --stop-at-task ${stop_at_task} \
    --approach supcon \
    --temperature ${temperature} \
    --num-exemplars ${exemplars} \
    --head-init-mode ${head_init}
fi
