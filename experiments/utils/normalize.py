from sentence_transformers import SentenceTransformer
import glob
import json
import re
import numpy as np

def getRelationsOf(ontology_path: str) -> list[str]:
    rels = []
    with open(ontology_path, "r") as f:
        ont = json.load(f)
        rel_objs = ont["relations"]
        for rel_obj in rel_objs:
            split = re.sub('([A-Z][a-z]+)', r' \1', re.sub('([A-Z]+)', r' \1', rel_obj["label"])).split()
            rels.append(" ".join(split).lower())
    return rels

def getClosestRelationOf(llm_relation: str, ont_relations: list[str], model: SentenceTransformer) -> str:
    llm_relation_embedding = model.encode(llm_relation)
    similarities: list[float] = []
    for ont_relation in ont_relations:
        ont_relation_embedding = model.encode(ont_relation)
        similarities.append(model.similarity(llm_relation_embedding, ont_relation_embedding))
    
    print("Closest relation in ont to", llm_relation, "is", ont_relations[np.argmax(similarities)])
    return ont_relations[np.argmax(similarities)]
        
        
def normalize_responses(llm_response_file_path: str, ont_path: str, model: SentenceTransformer):
    with open(llm_response_file_path, "r") as f_response:
        ont_relations = getRelationsOf(ont_path)
        response_data = [json.loads(line) for line in f_response]
        for response in response_data:
            for fact in response["triples"]:
                fact["rel"] = getClosestRelationOf(fact["rel"], ont_relations, model)
            
            with open("/".join(llm_response_file_path.split("/")[:-2]) + "/" + llm_response_file_path.split("/")[-2] + ".normalized/" + llm_response_file_path.split("/")[-1], "a") as out_f:
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

    model = SentenceTransformer('sentence-transformers/all-mpnet-base-v2', device="cuda")
    for ontology_name in dpedia_webnlg_files:
    
        #response_file_paths = glob.glob("../results/llm_responses/" + model_name + "/ont_*.jsonl")
        ont_path = "../../data/dpedia_webnlg/ontologies/" + ontology_name + ".json"
        response_file_path = "../../experiments/results/llm_responses/Babelscape.rebel-large/"+ ontology_name + "-test_dpedia_webnlg-train_dpedia_webnlg-n_examples_1.jsonl"
        normalize_responses(response_file_path, ont_path, model)
    