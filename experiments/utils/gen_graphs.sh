#!/bin/bash

title_model_names=(
    "GPT-3.5-Turbo, Prompt Crafting" 
    "GPT-4o, Prompt Crafting" 
    "REBEL-large, multiple beams"
    "REBEL-large, multiple beams, only ontology relations"
    "REBEL-large, multiple beams, relational mapping t=0.3"
    "REBEL-large, multiple beams, sentence entailement t=0.55"
    "REBEL-large, 12 beams, relation mapping, various thresholds"
    "REBEL-large, 12 beams, sentence entailement, various thresholds"
)

model_name_globs=(
    "gpt-3.5-turbo-*-shot" 
    "gpt-4o-*-shot" 
    "Babelscape.rebel-large-*-beams"
    "Babelscape.rebel-large-*-beams-rel-in-ontology"
    "Babelscape.rebel-large-*-beams-rel-map"
    "Babelscape.rebel-large-*-beams-entail"
    "Babelscape.rebel-large-12-beams-rel-map-t=*"
    "Babelscape.rebel-large-12-beams-entail-t=*"
)

# of length title_model_names, every model has a different ylims per metric
f1_ylims=(1.0 1.0 0.5 0.5 0.5 0.5 0.5 0.5)
precision_ylims=(1.0 1.0 0.3 0.3 0.3 0.3 0.3 0.3)
recall_ylims=(1.0 1.0 0.7 0.7 0.7 0.7 0.7 0.7)

MODEL_NAME_ITER=0
for title_model_name in "${title_model_names[@]}"
do
    for metric in avg_f1 avg_recall avg_precision avg_sub_halluc avg_obj_halluc avg_rel_halluc
    do
        for mode in mean median
        do

            ylim=0
            vicuna_f1_bars='False'
            if [[ $metric == "avg_f1" ]]; then
                ylim=${f1_ylims[MODEL_NAME_ITER]}
                if [[ $mode == "mean" ]]; then
                    vicuna_f1_bars='True'
                fi
            elif [[ $metric == "avg_recall" ]]; then
                ylim=${recall_ylims[MODEL_NAME_ITER]}

            elif [[ $metric == "avg_precision" ]]; then
                ylim=${precision_ylims[MODEL_NAME_ITER]}
            fi

            echo $title_model_name ${model_name_globs[MODEL_NAME_ITER]} $metric $mode $ylim $MODEL_NAME_ITER
            python3 bar_plots.py \
            --title_model_name  "$title_model_name" \
            --model_name_glob ${model_name_globs[MODEL_NAME_ITER]} \
            --metric $metric \
            --mode $mode \
            --ylim $ylim \
            --vicuna_f1_bars "$vicuna_f1_bars"
        done
    done
    MODEL_NAME_ITER=$(expr $MODEL_NAME_ITER + 1)
done
