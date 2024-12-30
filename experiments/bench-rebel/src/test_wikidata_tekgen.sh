#!/bin/bash

echo "Run test_wikidata_tekgen.sh"

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
    35.06
    41.35
    76.35
    46.86
    56.66
    51.63
    74.63
    63.08
    65.05
    74.25
)

module load GCCcore/13.2.0 Python/3.11.5

for i in "${!ontologies[@]}"
do
    echo $i
    sbatch --partition shared-gpu --ntasks 1 --mem 25G --time 2:00:00 --gres gpu:1,VramPerGpu:24G pipenv run python3 test.py data=wikidata_synthetic do_test_predict=True ontology_paths=[data/wikidata_tekgen/ontologies/${ontologies[$i]}.json] test_files=[data/wikidata_tekgen/test/${ontologies[$i]}_test.jsonl] checkpoint_path=experiments/bench-rebel/src/checkpoints/30-dec/${ontologies[$i]}-val_F1_micro\\=${best_val_f1s[$i]}.ckpt output_file_path=/experiments/results/llm_responses/rebel-fine-tuned-per-ontology-30-dec-checkpoints/${ontologies[$i]}-val_F1_micro\\=${best_val_f1s[$i]}/${ontologies[$i]}-wikidata_tekgen.jsonl
done
