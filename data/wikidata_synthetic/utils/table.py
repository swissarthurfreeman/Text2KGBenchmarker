#####################################################
# Script to get basic dataset statistics, number    #
# of samples per traint, validation and test folds. #
#####################################################

import json
import numpy as np

ontologies = [
    "ont_1_movie",
    "ont_2_music",
    "ont_3_sport",
    "ont_4_book",
    "ont_5_military",
    "ont_6_computer",
    "ont_7_space",
    "ont_8_politics",
    "ont_9_nature",
    "ont_10_culture"
]

print("Ontology, n°triples, median, p25, p75")
for ont in ontologies:

    n_samples = 0
    n_triples = 0
    n_triples_list = []
    

    for path in [f"../train/{ont}_train.jsonl", f"../validation/{ont}_val.jsonl", f"../test/{ont}_test.jsonl"]:
            
        with open(path) as f:
            data = [json.loads(line) for line in f]
            for sample in data:
                n_triples_list.append(len(sample['triples']))
                n_triples += len(sample['triples'])
            
            n_samples += len(data)

    words = ont.split("_")[1:]
    words[-1] = words[-1].capitalize()

    print(",".join([". ".join(words), str(n_samples), str(n_triples), str(np.percentile(n_triples_list, 25)), str(np.median(n_triples_list)), str(np.percentile(n_triples_list, 75))]))