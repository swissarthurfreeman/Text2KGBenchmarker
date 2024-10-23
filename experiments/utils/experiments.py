from adapter import OpenAIAdapter
from dotenv import load_dotenv
from run import LLMRunConfig
from run import LLMRun
import glob
import os

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

if __name__ == "__main__":
    load_dotenv("../../.env")
    
    for ontology_name in dpedia_webnlg_files:
        for n_examples in [1, 2, 3, 4, 5, 6]:
            conf = LLMRunConfig(
                "../../data/dpedia_webnlg/train/" + ontology_name + "_train.jsonl",
                "../../data/dpedia_webnlg/test/" + ontology_name + "_test.jsonl",
                "../../data/dpedia_webnlg/ontologies/" + ontology_name + ".json",
                n_examples,
                OpenAIAdapter(os.getenv('OPEN_API_KEY'), 'gpt-4o')
            )
            
            runner = LLMRun(conf)
            runner.run()
        
    