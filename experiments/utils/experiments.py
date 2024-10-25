from adapter import OpenAIAdapter, RebelAdapter
#from dotenv import load_dotenv
from run import LLMRunConfig
from run import LLMRun

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
    
    for ontology_name in wikidata_tekgen_files:
        conf = LLMRunConfig(
            "../../data/wikidata_tekgen/train/" + ontology_name + "_train.jsonl",
            "../../data/wikidata_tekgen/test/" + ontology_name + "_test.jsonl",
            "../../data/wikidata_tekgen/ontologies/" + ontology_name + ".json",
            1,
            RebelAdapter("Babelscape/rebel-large")
        )
        
        runner = LLMRun(conf)
        runner.run()
        
    