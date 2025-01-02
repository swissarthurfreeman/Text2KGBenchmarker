import torch
import glob
import json
from transformers import pipeline

def naive_sub_not_in_sent_ratio_of(path: str) -> None:
    files = glob.glob(path)

    res = []
    for file in files:
        data = []
        
        with open(file) as f:
            data = [json.loads(line) for line in f]
        
        n_no_subject_in_sent = 0
        n_triples = sum([len(sample['triples']) for sample in data]) 

        for sample in data:
            for triple in sample['triples']:
                if sample['sent'].lower().replace(" ", "").find(triple['sub'].lower().replace(" ", "")) == -1:
                    n_no_subject_in_sent += 1
        
        res.append((file.split('/')[-1], n_no_subject_in_sent/n_triples))

    res.sort(key=lambda tup: tup[1])

    for ont, val in res:
        print("{:.2f}".format( (val) * 100) + "%, is the percentage of triples with subject not in sentence in ontology", ont)

def entailed_triple_ratios_of(path: str, entailer) -> None:
    files = glob.glob(path)

    res = []
    for file in files:
        ont = file.split('/')[-1]
        data = []
        with open(file) as f:
            data = [json.loads(line) for line in f]

        n_triples_entailed = 0
        n_triples = sum([len(sample['triples']) for sample in data]) 
        
        for sample in data:
            for triple in sample['triples']:
                entailement_result = entailer( sample['sent'] + f". {triple['sub']} {triple['rel']} {triple['obj']}" )[0]
                #print(sample['sent'] + f". {triple['sub']} {triple['rel']} {triple['obj']}", entailement_result)

                with open(f"./{ont}.out", "a") as f:
                    f.write(sample['sent'] + f". {triple['sub']} {triple['rel']} {triple['obj']} " + entailement_result['label'] + "\n")
                
                if entailement_result["label"] == "ENTAILMENT":
                    n_triples_entailed += 1
        
        res.append((ont, n_triples_entailed/n_triples))

    res.sort(key=lambda tup: tup[1])

    for ont, val in res:
        print("{:.2f}".format( (val) * 100) + "%, is the percentage of triples entailed by sentence in ontology", ont)


if __name__ == '__main__':
    device = "cuda" if torch.cuda.is_available() else "cpu"
    sent_entailement_model = pipeline(model='roberta-large-mnli', device=device)



    for variant in ['wikidata_tekgen', 'dpedia_webnlg_clean']:
        entailed_triple_ratios_of(f'../../data/{variant}/test/*.jsonl', sent_entailement_model)
        #naive_sub_not_in_sent_ratio_of(f'../../data/{variant}/test/*.jsonl')
        
