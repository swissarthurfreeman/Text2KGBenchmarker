#!/bin/bash

module load GCCcore/13.2.0 Python/3.11.5

for t in 0.05 0.10 0.15 0.20 0.25 0.30 0.35 0.40 0.45 0.50 0.55 0.60 0.65 0.70 0.75 0.80 0.80 0.85 0.90 0.95 
do
    sbatch --partition shared-gpu --ntasks 1 --mem 25G --time 1:00:00 --gres gpu:1,VramPerGpu:24G pipenv run python3 normalize.py 12 $t 
done