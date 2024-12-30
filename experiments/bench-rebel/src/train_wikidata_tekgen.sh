#!/bin/bash

echo "Run train_wikidata_tekgen.sh" 

declare -a ontologies=(
    ont_1_movie
    ont_2_music
    ont_3_sport
    ont_4_book
    ont_5_military
    ont_6_computer
    ont_7_space
    ont_8_politics
    ont_9_nature
    ont_10_culture
)

module load GCCcore/13.2.0 Python/3.11.5

for ont in "${ontologies[@]}"
do
    echo $ont
    sbatch --partition shared-gpu --ntasks 1 --mem 25G --time 2:00:00 --gres gpu:1,VramPerGpu:24G --job-name $ont pipenv run python3 train.py data=wikidata_synthetic wandb_run_name=${ont}-wikidata-tekgen-train-and-val ontology_paths=[data/wikidata_tekgen/ontologies/${ont}.json] val_files=[data/wikidata_tekgen/validation/${ont}_validation.jsonl] train_files=[data/wikidata_tekgen/train/${ont}_train.jsonl]
done
