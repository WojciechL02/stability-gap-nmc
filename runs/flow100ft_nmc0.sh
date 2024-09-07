#!/bin/bash

#SBATCH --time=24:00:00   # walltime
#SBATCH --ntasks=8   # number of processor cores (i.e. tasks)
#SBATCH --gpus=1

set -e

eval "$(conda shell.bash hook)"
conda activate FACIL

num_tasks=10
nc_per_task="12 10 10 10 10 10 10 10 10 10"
stop_at_task=0  # default = 0
dataset=flowers
network=resnet18
tag=slca  # experiment name

num_epochs=30
bsz=128
lr=0.1
# exemplars=10
head_init=zeros

# for seed in 0 1 2; do
./experiments/ft_nmc_pretr.sh 0 0 ${tag} ${dataset} ${num_tasks} "${nc_per_task}" ${network} ${num_epochs} ${lr} ${head_init} ${stop_at_task} 1 5 ${bsz}
./experiments/ft_nmc_pretr.sh 0 0 ${tag} ${dataset} ${num_tasks} "${nc_per_task}" ${network} ${num_epochs} ${lr} ${head_init} ${stop_at_task} 1 40 ${bsz}
# ./experiments/ft_nmc.sh 0 0 ${tag} ${dataset} ${num_tasks} ${nc_first_task} ${network} 100 0.1 ${head_init} ${stop_at_task} 0 2000 &
# done
# wait

# exemplars=50
# for seed in 0 1 2; do
# ./experiments/ft_nmc.sh 0 0 ${tag} ${dataset} ${num_tasks} ${nc_first_task} ${network} ${num_epochs} ${lr} ${head_init} ${stop_at_task} 1 50
# done
# wait
