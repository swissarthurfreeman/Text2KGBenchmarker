from adapter import OpenAIAdapter
from dotenv import load_dotenv
from run import LLMRunConfig
from run import LLMRun
import glob
import os

if __name__ == "__main__":
    load_dotenv("../../.env")
    
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
    
    for ontology_name in wikidata_tekgen_files:
        conf = LLMRunConfig(
            "../../data/wikidata_tekgen/train/" + ontology_name + "_train.jsonl",
            "../../data/wikidata_tekgen/test/" + ontology_name + "_test.jsonl",
            "../../data/wikidata_tekgen/ontologies/" + ontology_name + ".json",
            1,
            OpenAIAdapter(os.getenv('OPEN_API_KEY'), 'gpt-4o')
        )
        
        runner = LLMRun(conf)
        runner.run()