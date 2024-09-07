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
wu_wd=${11:-0.0}
lr=${12:-0.1}
head_init=${13}
stop_at_task=${14:-0}
exemplars=${15:-20}
bsz=${16:-128}

if [ ${wu_epochs} -gt 0 ]; then
  exp_name="t${num_tasks}s20_wu_hz_wd:${wu_wd}_m:${exemplars}"
  result_path="results/${tag}/ft_wu_hz_${seed}"
  python3 src/main_incremental.py \
    --exp-name ${exp_name} \
    --gpu ${gpu} \
    --datasets ${dataset} \
    --num-tasks ${num_tasks} \
    --nc-first-task ${nc_first_task} \
    --network ${network} \
    --use-test-as-val \
    --lr ${lr} \
    --wu-wd ${wu_wd} \
    --nepochs ${num_epochs} \
    --batch-size ${bsz} \
    --seed ${seed} \
    --log disk wandb \
    --results-path ${result_path} \
    --tags ${tag} \
    --stop-at-task ${stop_at_task} \
    --approach finetuning \
    --scheduler-milestones \
    --num-exemplars-per-class ${exemplars} \
    --wu-nepochs ${wu_epochs} \
    --wu-lr ${wu_lr} \
    --wu-fix-bn \
    --wu-scheduler cosine \
    --head-init-mode ${head_init} \
    --pretrained \
    --slca
else
  exp_name="t${num_tasks}s20_hz_m:${exemplars}"
  result_path="results/${tag}/ft_hz_${seed}"
  python3 src/main_incremental.py \
    --exp-name ${exp_name} \
    --gpu ${gpu} \
    --datasets ${dataset} \
    --num-tasks ${num_tasks} \
    --nc-first-task ${nc_first_task} \
    --network ${network} \
    --use-test-as-val \
    --lr ${lr} \
    --wu-wd ${wu_wd} \
    --nepochs ${num_epochs} \
    --batch-size ${bsz} \
    --seed ${seed} \
    --log disk wandb \
    --results-path ${result_path} \
    --tags ${tag} \
    --scheduler-milestones \
    --stop-at-task ${stop_at_task} \
    --approach finetuning \
    --num-exemplars-per-class ${exemplars} \
    --head-init-mode ${head_init} \
    --pretrained \
    --slca
fi
