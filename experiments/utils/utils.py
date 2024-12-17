import re
import json

DPEDIA_WEBNLG_ONT_NAMES = [
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

WIKIDATA_TEKGEN_ONT_NAMES = [
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


def load_jsonl_as_dict(path: str) -> dict[str, dict[str, bool|list[str]|list[dict[str, str]]]]:
    with open(path, "r") as f:
        sent_dicts = [json.loads(line) for line in f]
        
        res = {}
        for sent_dict in sent_dicts:
            res[sent_dict['id']] = {}
            for key in sent_dict:
                res[sent_dict['id']][key] = sent_dict[key]
                
        return res

def load_jsonl_as_list(path: str) -> list[dict]:
    with open(path, "r") as f:
        return [json.loads(line) for line in f]
    
    
def getOntologyConceptsList(ontology_name: str, dataset_name: str) -> list[str]:
    """return list of concepts, concept surface forms words are seperated by spaces, splitting on camel case and lowercased."""
    with open("../../data/" + dataset_name + "/ontologies/" + ontology_name + ".json", "r") as ont_f:
        onto = json.load(ont_f)
        res = []
        for concept in onto["concepts"]:
            res.append(" ".join(camelCaseToSpaces(concept["label"]).split()).lower().strip())
        return res
    
def getOntologyRelationsList(ontology_name: str, dataset_name: str) -> list[str]:
    """return list of relations, with surface forms words seperated by spaces, splitting on camel case and lowercased.
    given relations: startedInYear, architect, return -> ['started in year', 'architect']. 
    """
    with open("../../data/" + dataset_name + "/ontologies/" + ontology_name + ".json", "r") as ont_f:
        onto = json.load(ont_f)
        
        concepts: dict[str, str] = {}
        for concept in onto['concepts']:
                concepts[concept['qid']] = concept['label']
                
        res = []
        for relation in onto["relations"]:
            rel_label = " ".join(camelCaseToSpaces(relation["label"]).lower().strip().replace(" ", "_").split())
            domain_label = concepts[relation['domain']]
            if relation['range'] == '': 
                range_label = 'literal'
            else:
                range_label = concepts[relation['range']]
                
            res.append(f"{rel_label}({domain_label} | {range_label})")
        return res

def camelCaseToSpaces(word: str) -> str:
    """Split camel cases to spaces, e.g. 'CamelCaseString Hello hello' -> '  Camel  Case  String   Hello hello'
    all this function does is replace every capital letter 'X' by ' X'."""
    return re.sub('([A-Z][a-z]+)', r' \1', re.sub('([A-Z]+)', r' \1', word))

