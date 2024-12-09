#!/bin/bash

#SBATCH --partition shared-cpu 
#SBATCH --ntasks 64 
#SBATCH --mem 25G 
#SBATCH --time 10:00:00  

module load GCCcore/13.2.0 Python/3.11.5

srun pipenv run python3 gen_triples.py Q5 ont_8_politics