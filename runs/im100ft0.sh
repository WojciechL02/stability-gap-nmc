#!/bin/bash

#SBATCH --time=24:00:00   # walltime
#SBATCH --ntasks=8   # number of processor cores (i.e. tasks)
#SBATCH --gpus=1

set -e

eval "$(conda shell.bash hook)"
conda activate FACIL

num_tasks=10
nc_first_task=10
stop_at_task=0  # default = 0
dataset=imagenet_subset_kaggle
network=resnet18
tag=figure1  # experiment name

num_epochs=100
lr=0.1
bsz=256
wu_lr=0.1
head_init=zeros
exemplars=2000

# without warm-up:
# for seed in 0 1 2; do
./experiments/ft2.sh 0 0 ${tag} ${dataset} ${num_tasks} ${nc_first_task} ${network} ${num_epochs} 0 0 0.0 ${lr} ${head_init} ${stop_at_task} ${exemplars} ${bsz}
# done
# wait
