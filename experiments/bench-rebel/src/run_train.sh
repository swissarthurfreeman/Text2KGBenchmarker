#!/bin/bash

#SBATCH --partition shared-gpu 
#SBATCH --ntasks 1 
#SBATCH --mem 25G 
#SBATCH --time 30:00 
#SBATCH --gres gpu:1,VramPerGpu:24G 

module load GCCcore/13.2.0 Python/3.11.5

srun pipenv run python3 train.py data=wkdata_synth_movie