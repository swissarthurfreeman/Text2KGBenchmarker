import re
import os
import json
import glob
import torch
import numpy as np
from transformers import pipeline
from sentence_transformers import SentenceTransformer
from utils import WIKIDATA_TEKGEN_ONT_NAMES, DPEDIA_WEBNLG_ONT_NAMES, load_jsonl_as_list, load_jsonl_as_dict

# TODO : rename output fiels to format below, change code in prompter and in rest of files to deal with this
# rigid file format.
# ll_response files always follow this format : /model_name-n-shot_specifier?/ontname-datasetname.jsonl and are always evaluated on
# the test data.
def getOntologyNameFrom(path: str) -> str:
        return path.split("/")[-1].split("-")[0]
    
def getDatasetNameFrom(path: str) -> str:
    return path.split("/")[-1].split("-")[1][:-6]


class Normalizer:
    """
    Helper interface to read all responses by LLM on files in `llm_responses/model_name/*.jsonl`, 
    create a new folder in `llm_results/model_name_normalized/` with the normalized responses. 
    Normalization means can mean replacing the relation by closest relation in ontology and 
    or filtering out response triples based on relation quality and entailment from the raw text.
    Exact implementation logic is left to the user in the `normalize(triple: dict[str, str])` method.
    """
    def __init__(self, model_to_normalize_name: str, normalized_model_name: str):
        """Normalizer('gpt-4o', 'gpt-4o-normalized')"""
        self.test_responses_paths: list[str] = glob.glob("../results/llm_responses/" + model_to_normalize_name + "/*.jsonl")
        self.model_to_normalize_name = model_to_normalize_name
        self.normalized_model_name = normalized_model_name
        self.normalized_llm_responses_dir = "../results/llm_responses/" + self.normalized_model_name
        
        if not os.path.exists(self.normalized_llm_responses_dir):
            os.makedirs(self.normalized_llm_responses_dir)
    
    # TODO : get rid of this function call
    def getRelationsListOfDictsOf(self, dataset_name: str, ontology_name: str) -> list[dict[str, str]]:
        rels = []
        with open("../../data/" + dataset_name + "/ontologies/" + ontology_name + ".json", "r") as f:
            ont = json.load(f)
            rel_objs = ont["relations"]
            for rel_obj in rel_objs:
                split = re.sub('([A-Z][a-z]+)', r' \1', re.sub('([A-Z]+)', r' \1', rel_obj["label"])).split()
                rels.append({"rel": " ".join(split).lower(), "domain": rel_obj["domain"], "range": rel_obj["range"]})
        return rels

    def normalize(self, response: dict, sentence: str, ont_relations: list[dict[str, str]]) -> dict:
        """Normalize all triples in repsonse, this method can drop triples."""
        raise NotImplementedError
    
    def generateNormalizedData(self):
        for response_path in self.test_responses_paths:
            responses: list[dict] = load_jsonl_as_list(response_path)
            dataset_name, ontology_name = getDatasetNameFrom(response_path), getOntologyNameFrom(response_path)
            
            ont_relations: list[dict[str, str]] = self.getRelationsListOfDictsOf(dataset_name, ontology_name)
            
            ground_truths = load_jsonl_as_dict("../../data/" + dataset_name + "/test/" + ontology_name + "_test.jsonl")
            
            normalized_responses = []
            for response in responses:
                sentence = ground_truths[response["id"]]["sent"]
                normalized_responses.append(self.normalize(response, sentence, ont_relations))
                
            for normalized_response in normalized_responses:
                with open(self.normalized_llm_responses_dir + "/" + ontology_name + "-" + dataset_name + ".jsonl", "w") as f_out:
                    f_out.write(json.dumps(normalized_response) + "\n")
            

class Similarity(Normalizer):
    """Replace all relation surface forms of the llm responses by the closest in the ontology, drop
    triple if maximum similarity is underneath the threshold."""
    def __init__(self, model_to_normalize_name: str, normalized_model_name: str, threshold: float, sent_embedder: SentenceTransformer):
        super().__init__(model_to_normalize_name, normalized_model_name)
        self.sent_embedder = sent_embedder
        self.threshold = threshold
    
    def getClosestOntologyRelationTo(self, llm_relation: dict[str, str], ont_relations: list[dict[str, str]]) -> dict[str, str] | None:
        llm_relation_embedding = self.sent_embedder.encode(" ".join([llm_relation["sub"], llm_relation["rel"], llm_relation["obj"]]))
        similarities: list[float] = []
        
        for ont_relation in ont_relations:
            ont_relation_embedding = self.sent_embedder.encode(" ".join([ont_relation["domain"], ont_relation["rel"], ont_relation["range"]]))
            similarities.append(self.sent_embedder.similarity(llm_relation_embedding, ont_relation_embedding))
        
        #print("Closest relation to", prettyString(llm_relation), "is", prettyString(ont_relations[np.argmax(similarities)]), "similarity :", max(similarities).item())
        if max(similarities) < self.threshold: return None    # if low confidence, don't bother
        return ont_relations[np.argmax(similarities)]
    
    def normalize(self, response: dict, sentence: str, relations: list[dict[str, str]]) -> dict:
        """Normalize all triples in response, this method can drop triples."""
        res = {"id": response["id"], "response": response["response"], "triples": []}
        
        for triple in response["triples"]:
            closest_rel: dict[str, str] = self.getClosestOntologyRelationTo(triple, relations)    # {"sub": "human", "rel": "screenwriter", "obj": "film"}
            if closest_rel != None:
                triple["rel"] = closest_rel["rel"]
                res["triples"].append(triple)
        return response
        

class Entailement(Normalizer):
    """Drop triple if it is not entailed by the sentence"""
    def __init__(self, model_to_normalize_name: str, normalized_model_name: str, sent_entailer: object, threshold: float):
        super().__init__(model_to_normalize_name, normalized_model_name)
        self.sent_entailer = sent_entailer
        self.neutral_entailment_threshold = threshold
    
    def normalize(self, response: dict, sentence: str, relations: list[dict[str, str]]) -> dict:
        """Normalize all triples in response, this method can drop triples."""
        res = {"id": response["id"], "response": response["response"], "triples": []}
        
        for triple in response["triples"]:
            entailement_result = self.sent_entailer(sentence)[0]
            # if sentence entails result, or we have reasonable doubt it might, keep triple.
            if entailement_result["label"] == "ENTAILMENT" or (res["label"] == "NEUTRAL" and res["score"] < self.neutral_entailment_threshold):
                res["triples"].append(triple)
        return res
            
if __name__ == "__main__":    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    sent_comp_model = SentenceTransformer('sentence-transformers/all-mpnet-base-v2', device=device)
    sim_normalizer = Similarity("Babelscape.rebel-large", "Babelscape.rebel-large.normalized-similarity", 0.3, sent_comp_model)
    sim_normalizer.generateNormalizedData()
    
    # you can then add another layer of normalization via, 
    sent_entailement_model = pipeline(model='roberta-large-mnli', device=device)
    ent_normalizer = Entailement("Babelscape.rebel-large.normalized-similarity", "Babelscape.rebel-large.normalized-similarity-entailement", sent_entailement_model, 0.55)
    ent_normalizer.generateNormalizedData()