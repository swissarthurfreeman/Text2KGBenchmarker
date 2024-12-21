import json

onto_name = "ont_10_culture"
data_split = "train"

with open(f"{onto_name}_{data_split}.jsonl", "r") as f:
    data = [json.loads(line) for line in f]
    
    res = {}
    for line in data:
        res[line['triples'][0]['sqid']] = {
            'id': f'{onto_name}_{data_split}_{line["triples"][0]["sqid"]}',
            'sent': line['sent'],
            'triples': line['triples']   
        }
        
    for value in res.values():
        with open(f"{onto_name}_{data_split}_clean.jsonl", "a") as g:
            g.write(json.dumps(value) + "\n")
        
        
        