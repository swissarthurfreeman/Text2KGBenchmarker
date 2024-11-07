# again, we need to fold the validation data, duplicate sentences, and quality is highly questionable
# it might be worth manually re-annotating a single ontology like the movie one. 

import json
import glob

files = glob.glob("*.jsonl")

for path in files:
    with open(path) as f:
        data = [json.loads(line) for line in f]
        res = {}
        
        for sent in data:
            if sent["sent"] not in res.keys():
                res[sent["sent"]] = {
                    "id": sent["id"],
                    "sent": sent["sent"],
                    "triples": [{
                        "sub": sent["sub_label"], 
                        "rel": sent["rel_label"], 
                        "obj": sent["obj_label"]
                    }]
                }
            else:
                res[sent["sent"]]["triples"].append({
                    "sub": sent["sub_label"],
                    "rel": sent["rel_label"],
                    "obj": sent["obj_label"]
                })
        
        with open(path + "_clean.jsonl", "a") as g:
            for val in res.values():
                g.write(json.dumps(val) + "\n")