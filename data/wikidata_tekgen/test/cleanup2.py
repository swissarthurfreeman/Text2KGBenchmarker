import json
import glob
from pprint import pprint

test_files = glob.glob("./*.jsonlp")

for test_file in test_files:
    with open(test_file, "r") as sent_f:
        with open("../train/" + test_file[2:-12] + "_train.jsonlp_folds", "r") as folds_f:
            with open(test_file + "_folded_final", "w") as res_f:
                json_data = [json.loads(line) for line in sent_f]
                folds: dict[str, str] = json.load(folds_f)
                
                for sentence in json_data:
                    for idx in range(len(sentence["similars"])):
                        if sentence["similars"][idx] in folds.keys():
                            sentence["similars"][idx] = folds[sentence["similars"][idx]]
                    
                    sentence["similars"] = list(set(sentence["similars"]))
                    res_f.write(json.dumps(sentence) + "\n")