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
lr=0.1
bsz=128
wu_lr=0.1
head_init=zeros
# exemplars=50

# without warm-up:
# for seed in 0 1 2; do
./experiments/ft_pretr.sh 0 2 ${tag} ${dataset} ${num_tasks} "${nc_per_task}" ${network} ${num_epochs} 0 0 0.0 ${lr} ${head_init} ${stop_at_task} 5 ${bsz}
# done
# wait