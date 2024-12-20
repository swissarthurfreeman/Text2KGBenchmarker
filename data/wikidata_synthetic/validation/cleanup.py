import json

with open("ont_2_music_val.jsonl", "r") as f:
    data = [json.loads(line) for line in f]
    
    res = {}
    for line in data:
        res[line['triples'][0]['sqid']] = {
            'id': f'ont_2_music_train_{line["triples"][0]["sqid"]}',
            'sent': line['sent'],
            'triples': line['triples']   
        }
        
    for value in res.values():
        with open("ont_2_music_train_clean.jsonl", "a") as g:
            g.write(json.dumps(value) + "\n")
        
        
        