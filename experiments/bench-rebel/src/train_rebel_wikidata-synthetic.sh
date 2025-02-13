#!/bin/bash

echo "Running fine-tune of REBEL training using synthetic Wikidata-TekGen data and original data as validation."

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

echo "Run fine-tune for Wikidata-TekGen using only Wikidata-Synthetic as training data." 

for ont in "${ontologies[@]}"
do
    echo $ont
    sbatch --partition shared-gpu --ntasks 1 --mem 25G --time 4:00:00 --gres gpu:1,VramPerGpu:24G --job-name $ont \
    pipenv run python3 train.py \
    data=text2kgbench-fine-tune \
    max_steps=2000 \
    wandb_project_name=Text2KGBench-Wikidata-Synthetic-fine-tune \
    wandb_run_name=${ont}-Wikidata-Synthetic-train-TekGen-val \
    ontology_paths=[data/wikidata_synthetic/ontologies/${ont}.json] \
    train_files=[data/wikidata_synthetic/train/${ont}_train.jsonl] \
    val_files=[data/wikidata_tekgen/validation/${ont}_validation.jsonl]
done


echo "Run fine-tune for Wikidata-TekGen using Wikidata-TekGen+Synthetic training data."

for ont in "${ontologies[@]}"
do
    echo $ont
    sbatch --partition shared-gpu --ntasks 1 --mem 25G --time 4:00:00 --gres gpu:1,VramPerGpu:24G --job-name $ont \
    pipenv run python3 train.py \
    data=text2kgbench-fine-tune \
    max_steps=2000 \
    wandb_project_name=Text2KGBench-Wikidata-TekGen+Synthetic-fine-tune \
    wandb_run_name=${ont}-Wikidata-TekGen+Synthetic-train-TekGen-val \
    ontology_paths=[data/wikidata_synthetic/ontologies/${ont}.json] \
    train_files=[data/wikidata_tekgen/train/${ont}_train.jsonl,data/wikidata_synthetic/train/${ont}_train.jsonl] \
    val_files=[data/wikidata_tekgen/validation/${ont}_validation.jsonl]
done

