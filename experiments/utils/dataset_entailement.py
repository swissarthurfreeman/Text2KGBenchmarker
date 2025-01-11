import os
import glob
import json
import nltk
import torch
import itertools
from nltk import PorterStemmer
from transformers import pipeline
from utils import WIKIDATA_TEKGEN_ONT_NAMES, DPEDIA_WEBNLG_ONT_NAMES

def getEntailementRatioOf(samples: list[dict[str, str | list[dict[str, str]]]], variant: str, entailer) -> float:
    """Compute triple entailement ratio for list of samples where every sample has an `id`, `sent`, `triples` keys."""
    
    if not os.path.exists(f'../results/quality/{variant}/'):
        os.makedirs(f'../results/quality/{variant}/')

    n_triples_total = sum([len(sample['triples']) for sample in samples]) 
    n_triples_entailed = 0
    
    for sample in samples:
        for triple in sample['triples']:
            entailed = False
            sentences = nltk.sent_tokenize(sample['sent'])              # nltk will tokenize well, dealing with colons in sentences in abbreviations etc.

            for L in range(len(sentences) + 1):
                for subset in itertools.combinations(sentences, L):     # for every combination of sentences
                    sent = " ".join(subset)                             # stick them together, check if they entail the triple
                    result = entailer( sent + f". {triple['sub']} {triple['rel']} {triple['obj']}" )[0]
                    
                    with open(f"../results/quality/{variant}/{ont}.out", "a") as f:
                        f.write(json.dumps({
                            'id': sample['id'], 
                            'sent': sent, 
                            'triple': f"{triple['sub']} {triple['rel']} {triple['obj']}", 
                            'entailement': result["label"]
                        }) + "\n")

                    if result['label'] == 'ENTAILMENT':                 # if triple is entailed by at a combination
                        entailed = True                                 # no need to check other combinations, set flag and break from loop
                        break
                        
            if entailed:
                n_triples_entailed += 1

    return n_triples_entailed / n_triples_total

def getTitle(onto_name: str) -> str:
    """ont_1_movie -> 1. Movie"""
    parts = onto_name.split("_")
    return parts[1] + ". " + parts[-1].capitalize()

def getSamplesList(path: str, n: int = 500) -> list[dict]:
    samples = []
    with open(path) as f:
        samples = [json.loads(line) for line in f]
    return samples[:n]

def getDBpediaWebNLGEntailRatios(sent_entailement_model) -> None:
    for variant in ['dbpedia_webnlg_clean']:
        with open(f"../results/quality/entailements_{variant}.csv", 'a') as f:
            f.write("Ontology,Train,Test\n")    

        for ont in DPEDIA_WEBNLG_ONT_NAMES:
            with open(f"../results/quality/entailements_{variant}.csv", 'a') as f:
                f.write(getTitle(ont))
                for split in ['train', 'test']:
                    file_path = f'../../data/dpedia_webnlg_clean/{split}/{ont}_{split}.jsonl'
                    if os.path.exists(file_path):
                        samples = getSamplesList(file_path, 1000)
                        ratio = getEntailementRatioOf(samples, variant, sent_entailement_model)
                        f.write(","+"{:.2f}".format(ratio))
                f.write("\n")

def getWikidataTekGenEntailRatios(sent_entailement_model) -> None:
    for variant in ['wikidata_tekgen', 'wikidata_synthetic']:
        with open(f"../results/quality/entailements_{variant}.csv", "a") as f:
            f.write("Ontology,Train,Validation,Test\n")
            
        for ont in WIKIDATA_TEKGEN_ONT_NAMES:
            with open(f"../results/quality/entailements_{variant}.csv", "a") as f:
                f.write(getTitle(ont))

                for split in ['train', 'validation', 'test']:
                    name = split
                    if split == 'validation' and variant == 'wikidata_synthetic':  name = 'val'
                    file_path = f'../../data/{variant}/{split}/{ont}_{name}.jsonl'

                    if os.path.exists(file_path):        
                        samples = getSamplesList(file_path, 1000 if variant == 'wikidata_tekgen' else 500)
                        ratio = getEntailementRatioOf(samples, variant, sent_entailement_model)
                        f.write(","+"{:.2f}".format(ratio))
                    else:
                        print("path", f"../../data/{variant}/{split}/{ont}_{split}.jsonl")
                
                f.write("\n")
    
def getSubjectInSentRatioOf(samples: list[dict[str, str | list[dict[str, str]]]]) -> float:
    ps = PorterStemmer()
    n_subject_in_sent = 0
    n_triples = sum([len(sample['triples']) for sample in samples]) 

    for sample in samples:
        for triple in sample['triples']:
            if ps.stem(sample['sent']).lower().replace(" ", "").find(ps.stem(triple['sub']).lower().replace(" ", "")) != -1:
                n_subject_in_sent += 1
    
    return n_subject_in_sent/n_triples

def getDBpediaWebNLGSubjRatios() -> None:
    for variant in ['dbpedia_webnlg_clean']:
        with open(f"../results/quality/subj_ratios_{variant}.csv", 'a') as f:
            f.write("Ontology,Train,Test\n")    

        for ont in DPEDIA_WEBNLG_ONT_NAMES:
            with open(f"../results/quality/subj_ratios_{variant}.csv", 'a') as f:
                f.write(getTitle(ont))
                for split in ['train', 'test']:
                    file_path = f'../../data/dpedia_webnlg_clean/{split}/{ont}_{split}.jsonl'
                    if os.path.exists(file_path):
                        samples = getSamplesList(file_path, 1000)
                        ratio = getSubjectInSentRatioOf(samples)
                        f.write(","+"{:.2f}".format(ratio))
                f.write("\n")

def getWikidataTekGenSubjRatios() -> None:
    for variant in ['wikidata_tekgen', 'wikidata_synthetic']:
        with open(f"../results/quality/subj_ratios_{variant}.csv", "a") as f:
            f.write("Ontology,Train,Validation,Test\n")
        for ont in WIKIDATA_TEKGEN_ONT_NAMES:
            with open(f"../results/quality/subj_ratios_{variant}.csv", "a") as f:
                f.write(getTitle(ont))
                
                for split in ['train', 'validation', 'test']:
                    name = split
                    if split == 'validation' and variant == 'wikidata_synthetic':  name = 'val'
                    file_path = f'../../data/{variant}/{split}/{ont}_{name}.jsonl'

                    if os.path.exists(file_path):   
                        samples = getSamplesList(file_path, 1000 if variant == 'wikidata_tekgen' else 500)
                        ratio = getSubjectInSentRatioOf(samples)
                        f.write(","+"{:.2f}".format(ratio))
                    else:
                        print("path", f"../../data/{variant}/{split}/{ont}_{split}.jsonl")
                f.write("\n")


if __name__ == '__main__':
    #device = "cuda" if torch.cuda.is_available() else "cpu"
    #sent_entailement_model = pipeline("text-classification", model='roberta-large-mnli', device=device)

    getDBpediaWebNLGSubjRatios()
    getWikidataTekGenSubjRatios()

    #getWikidataTekGenEntailRatios(sent_entailement_model)
    #getDBpediaWebNLGEntailRatios(sent_entailement_model)
    