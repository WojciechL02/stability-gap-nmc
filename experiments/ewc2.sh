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
wu_epochs=${9:-0}
wu_lr=${10:-0.1}
wu_wd=${11:-0}
lr=${12:-0.1}
lamb=${13}
head_init=${14}
stop_at_task=${15:-0}

if [ ${wu_epochs} -gt 0 ]; then
  exp_name="cifar100t${num_tasks}s${nc_first_task}_${tag}_wu_hz_wd:${wu_wd}"
  result_path="results/${tag}/ewc_wu_hz_${seed}"
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
    --batch-size 128 \
    --seed ${seed} \
    --log disk wandb \
    --results-path ${result_path} \
    --tags ${tag} \
    --scheduler-milestones \
    --cm \
    --stop-at-task ${stop_at_task} \
    --approach ewc \
    --lamb ${lamb} \
    --wu-nepochs ${wu_epochs} \
    --wu-lr ${wu_lr} \
    --wu-wd ${wu_wd} \
    --wu-fix-bn \
    --wu-scheduler cosine \
    --head-init-mode ${head_init}
else
  exp_name="cifar100t${num_tasks}s${nc_first_task}_${tag}_hz"
  result_path="results/${tag}/ewc_hz_${seed}"
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
    --batch-size 128 \
    --seed ${seed} \
    --log disk wandb \
    --results-path ${result_path} \
    --tags ${tag} \
    --scheduler-milestones \
    --cm \
    --stop-at-task ${stop_at_task} \
    --approach ewc \
    --lamb ${lamb} \
    --head-init-mode ${head_init}
fi
