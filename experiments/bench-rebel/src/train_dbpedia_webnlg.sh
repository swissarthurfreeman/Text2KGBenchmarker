#!/bin/bash

echo "Run train_dbpedia_webnlg.sh" 

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
    sbatch --partition shared-gpu --ntasks 1 --mem 25G --time 2:00:00 --gres gpu:1,VramPerGpu:24G --job-name $ont pipenv run python3 train.py data=wikidata_synthetic wandb_run_name=${ont}-dbpedia-webnlg-train-train ontology_paths=[data/dbpedia_webnlg_clean/ontologies/${ont}.json] val_files=[data/dbpedia_webnlg_clean/test/${ont}_test.jsonl] train_files=[data/dbpedia_webnlg_clean/train/${ont}_train.jsonl]
done
