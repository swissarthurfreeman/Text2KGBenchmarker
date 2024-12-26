#!/bin/bash

module load GCCcore/13.2.0 Python/3.11.5

for i in 2 4 6 8 10 12 
do
    sbatch --partition shared-gpu --ntasks 1 --mem 25G --time 1:00:00 --gres gpu:1,VramPerGpu:24G pipenv run python3 normalize.py $i 
done