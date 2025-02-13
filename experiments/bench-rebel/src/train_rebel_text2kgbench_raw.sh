#!/bin/bash

echo "Running fine-tune of REBEL over original Text2KGBench data."
echo "Run fine-tune for Wikidata-TekGen." 

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
    sbatch --partition shared-gpu --ntasks 1 --mem 25G --time 2:00:00 --gres gpu:1,VramPerGpu:24G --job-name $ont \
    pipenv run python3 train.py \
    data=text2kgbench-fine-tune \
    wandb_project_name=Text2KGBench-Wikidata-TekGen-fine-tune \
    wandb_run_name=${ont}-Wikidata-TekGen-train-val \
    ontology_paths=[data/wikidata_tekgen/ontologies/${ont}.json] \
    val_files=[data/wikidata_tekgen/validation/${ont}_validation.jsonl] \
    train_files=[data/wikidata_tekgen/train/${ont}_train.jsonl]
done

echo "Run fine-tune for DBpedia-WebNLG" 

declare -a ontologies=(
    ont_1_university
    ont_2_musicalwork
    ont_3_airport
    ont_4_building
    ont_5_athlete
    ont_6_politician
    ont_7_company
    ont_8_celestialbody
    ont_9_astronaut
    ont_10_comicscharacter
    ont_11_meanoftransportation
    ont_12_monument
    ont_13_food
    ont_14_writtenwork
    ont_15_sportsteam
    ont_16_city
    ont_17_artist
    ont_18_scientist
    ont_19_film
)

module load GCCcore/13.2.0 Python/3.11.5

for ont in "${ontologies[@]}"
do
    echo $ont
    sbatch --partition shared-gpu --ntasks 1 --mem 25G --time 2:00:00 --gres gpu:1,VramPerGpu:24G --job-name $ont \
    pipenv run python3 train.py \
    data=text2kgbench-fine-tune \
    wandb_project_name=Text2KGBench-DBpedia-WebNLG-fine-tune \
    wandb_run_name=${ont}-DBpedia-WebNLG-train-test \
    ontology_paths=[data/dbpedia_webnlg_clean/ontologies/${ont}.json] \
    val_files=[data/dbpedia_webnlg_clean/test/${ont}_test.jsonl] \
    train_files=[data/dbpedia_webnlg_clean/train/${ont}_train.jsonl]
done

