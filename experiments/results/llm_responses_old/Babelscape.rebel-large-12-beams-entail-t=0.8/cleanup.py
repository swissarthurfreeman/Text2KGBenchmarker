import json
import glob

files = glob.glob("*.jsonl")

for file in files:
    with open(file, "r") as f:
        data = [json.loads(line) for line in f]
        
        res = {}
        for line in data:
            res[line['id']] = line
            
        for value in res.values():
            with open(file.split(".")[0] + "_clean.jsonl", "a") as g:
                g.write(json.dumps(value) + "\n")
        
        
        