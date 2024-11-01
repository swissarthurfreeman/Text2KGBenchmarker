import re
import os
import json
import glob

directories = glob.glob("rebel-large*")

for dir in directories:
    os.makedirs("./" + dir + "_clean")
    res_files = glob.glob("./" + dir + "/*.jsonl")
    
    for res_file in res_files:
        with open(res_file, "r") as f:
            responses = [json.loads(line) for line in f]
            clean_file_name = res_file.split("/")[-1].replace("test_dpedia_webnlg-train_dpedia_webnlg", "dpedia_webnlg").replace("test_wikidata_tekgen-train_wikidata_tekgen", "wikidata_tekgen")
            clean_file_name = clean_file_name.replace("-n_examples", "")[:-8] + ".jsonl"
            #clean_file_name = res_file.split("/")[-1]
            with open("./" + dir + "_clean/" + clean_file_name, "w") as g:
                for r in responses:
                    g.write(json.dumps(r) + "\n")
                
            