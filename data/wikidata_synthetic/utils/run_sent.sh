#!/bin/bash

#SBATCH --partition shared-cpu
#SBATCH --ntasks 4
#SBATCH --mem 10G 
#SBATCH --time 6:00:00 

module load GCCcore/13.2.0 Python/3.11.5

srun pipenv run python3 gen_sent.py