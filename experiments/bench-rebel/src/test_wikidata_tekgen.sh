#!/bin/bash

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

declare -a best_val_f1s=(
    20.64
    28.69
    76.16
    36.86
    47.22
    50.98
    73.00
    53.39
    63.68
    71.33
)

module load GCCcore/13.2.0 Python/3.11.5

for i in "${!ontologies[@]}"
do
    sbatch --partition shared-gpu --ntasks 1 --mem 25G --time 2:00:00 --gres gpu:1,VramPerGpu:24G pipenv run python3 test.py 'data=wikidata_synthetic' 'do_test_predict=True' ontology_paths=[data/wikidata_synthetic/ontologies/${ontologies[$i]}.json] test_files=[data/wikidata_tekgen/test/${ontologies[$i]}_test.jsonl] checkpoint_path=experiments/bench-rebel/src/checkpoints/${ontologies[$i]}-val_F1_micro\\=${best_val_f1s[$i]}.ckpt output_file_path=experiments/results/llm_responses/${ontologies[$i]}-val_F1_micro\\=${best_val_f1s[$i]}/${ontologies[$i]}-wikidata_tekgen.jsonl
done
