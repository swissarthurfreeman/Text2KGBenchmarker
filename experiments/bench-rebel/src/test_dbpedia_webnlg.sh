#!/bin/bash

echo "Run test_wikidata_tekgen.sh"

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

declare -a best_val_f1s=(
    24.38
    25.34
    42.08
    48.42
    55.77
    45.61
    26.30
    71.87
    39.60
    47.62
    37.37
    18.37
    61.79
    49.47
    36.06
    47.51
    44.40
    36.85
    25.56
)

module load GCCcore/13.2.0 Python/3.11.5

for i in "${!ontologies[@]}"
do
    echo $i
    sbatch --partition shared-gpu --ntasks 1 --mem 25G --time 2:00:00 --gres gpu:1,VramPerGpu:24G pipenv run python3 test.py data=wikidata_synthetic do_test_predict=True ontology_paths=[data/dbpedia_webnlg_clean/ontologies/${ontologies[$i]}.json] test_files=[data/dbpedia_webnlg_clean/test/${ontologies[$i]}_test.jsonl] checkpoint_path=experiments/bench-rebel/src/checkpoints/30-dec/${ontologies[$i]}-val_F1_micro\\=${best_val_f1s[$i]}.ckpt output_file_path=/experiments/results/llm_responses/rebel-fine-tuned-per-ontology-30-dec-checkpoints/${ontologies[$i]}-val_F1_micro\\=${best_val_f1s[$i]}/${ontologies[$i]}-dbpedia_webnlg_clean.jsonl
done
