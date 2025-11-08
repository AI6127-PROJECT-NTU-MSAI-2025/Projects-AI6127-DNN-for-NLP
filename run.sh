#!/bin/bash
#SBATCH --job-name=m2m100_finetune
#SBATCH --time=06:00:00
#SBATCH --output=m2m100_new.log

#SBATCH --partition=MGPU-TC2
#SBATCH --qos=normal
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=10
#SBATCH --mem=30G

python finetune_notationed.py