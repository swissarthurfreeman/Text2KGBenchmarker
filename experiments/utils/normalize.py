from sentence_transformers import SentenceTransformer
from transformers import pipeline
import glob
import json
import re
import numpy as np

def load_jsonl_as_dict(path: str) -> dict[str, dict[str, bool|list[str]|list[dict[str, str]]]]:
    with open(path, "r") as f:
        sent_dicts = [json.loads(line) for line in f]
        
        res = {}
        for sent_dict in sent_dicts:
            res[sent_dict['id']] = {}
            for key in sent_dict:
                res[sent_dict['id']][key] = sent_dict[key]
                
        return res


def getRelationsOf(ontology_path: str) -> list[dict[str]]:
    rels = []
    with open(ontology_path, "r") as f:
        ont = json.load(f)
        rel_objs = ont["relations"]
        for rel_obj in rel_objs:
            split = re.sub('([A-Z][a-z]+)', r' \1', re.sub('([A-Z]+)', r' \1', rel_obj["label"])).split()
            rels.append({"rel": " ".join(split).lower(), "domain": rel_obj["domain"], "range": rel_obj["range"]})
    return rels

def prettyString(relation: dict) -> str:
    if "sub" in relation.keys():
        return relation["rel"] + "(" + relation["sub"] + ", " + relation["obj"] + ")"
    return relation["rel"] + "(" + relation["domain"] + ", " + relation["range"] + ")"

def getClosestRelationOf(llm_relation: dict[str, str], ont_relations: list[dict[str]], model: SentenceTransformer) -> dict[str, str] | None:
    llm_relation_embedding = model.encode(" ".join([llm_relation["sub"], llm_relation["rel"], llm_relation["obj"]]))
    similarities: list[float] = []
    for ont_relation in ont_relations:
        ont_relation_embedding = model.encode(" ".join([ont_relation["domain"], ont_relation["rel"], ont_relation["range"]]))
        similarities.append(model.similarity(llm_relation_embedding, ont_relation_embedding))
    
    #print("Closest relation to", prettyString(llm_relation), "is", prettyString(ont_relations[np.argmax(similarities)]), "similarity :", max(similarities).item())
    if max(similarities) < 0.3: return None    # if low confidence, don't bother
    return ont_relations[np.argmax(similarities)]
        
        
def normalize_responses(llm_response_file_path: str, ont_path: str, model: SentenceTransformer, sent_entailement_model, test_data: dict):
    with open(llm_response_file_path, "r") as f_response:
        ont_relations: list[dict] = getRelationsOf(ont_path)
        response_data = [json.loads(line) for line in f_response]
        for response in response_data:
            triples = []
            for fact in response["triples"]:
                ont_rel: dict[str, str] = getClosestRelationOf(fact, ont_relations, model)
                if ont_rel != None:
                    fact["rel"] = ont_rel["rel"]   # keep only if we're sure the fact appears (should lower hallucinations)
                    
                    sentence = test_data[ response["sent_id"] ]["sent"] + " " + fact["sub"] + " " + fact["rel"] + " " + fact["obj"]
                    res = sent_entailement_model(sentence)[0]
                    
                    print("Sentence :", sentence, ", result :", res)
                    if res["label"] == "ENTAILMENT" or (res["label"] == "NEUTRAL" and res["score"] < 0.55):
                        triples.append(fact)
            
            print("Kept triples :", triples)
            response["triples"] = triples  
            
            with open("/".join(llm_response_file_path.split("/")[:-2]) + "/" + llm_response_file_path.split("/")[-2] + ".normalized-complex/" + llm_response_file_path.split("/")[-1], "a") as out_f:
                out_f.write(json.dumps(response) + "\n")
            
if __name__ == "__main__":    
    dpedia_webnlg_files = [
        "ont_1_university",
        "ont_2_musicalwork",
        "ont_3_airport",
        "ont_4_building",
        "ont_5_athlete",
        "ont_6_politician",
        "ont_7_company",
        "ont_8_celestialbody",
        "ont_9_astronaut",
        "ont_10_comicscharacter",
        "ont_11_meanoftransportation",
        "ont_12_monument",
        "ont_13_food",
        "ont_14_writtenwork",
        "ont_15_sportsteam",
        "ont_16_city",
        "ont_17_artist",
        "ont_18_scientist",
        "ont_19_film"
    ]

    wikidata_tekgen_files = [
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

    sent_comp_model = SentenceTransformer('sentence-transformers/all-mpnet-base-v2', device="cuda")
    sent_entailement_model = pipeline(model='roberta-large-mnli', device="cuda")
    
    """
    for ontology_name in wikidata_tekgen_files:
        ont_path = "../../data/wikidata_tekgen/ontologies/" + ontology_name + ".json"
        response_file_path = "../../experiments/results/llm_responses/Babelscape.rebel-large/"+ ontology_name + "-test_wikidata_tekgen-train_wikidata_tekgen-n_examples_1.jsonl"
        test_data = load_jsonl_as_dict("../../data/wikidata_tekgen/test/" + ontology_name + "_test.jsonl") 
        normalize_responses(response_file_path, ont_path, sent_comp_model, sent_entailement_model, test_data)
    """
    
    for ontology_name in dpedia_webnlg_files[8:]:
        ont_path = "../../data/dpedia_webnlg/ontologies/" + ontology_name + ".json"
        response_file_path = "../../experiments/results/llm_responses/Babelscape.rebel-large/"+ ontology_name + "-test_dpedia_webnlg-train_dpedia_webnlg-n_examples_1.jsonl"
        test_data = load_jsonl_as_dict("../../data/dpedia_webnlg/test/" + ontology_name + "_test.jsonl") 
        normalize_responses(response_file_path, ont_path, sent_comp_model, sent_entailement_model, test_data)
    