
class Prompter:
    """
    Allows to easily generate prompts from sentences with a description of
    the ontolgoy and a specified number of examples. 
    
    Usage
    -----
    prompter = Prompter(
        "../data/wikidata_tekgen/ontologies/1_movie_ontology.json"
        "../data/wikidata_tekgen/train/ont_1_movie_train.jsonl"
        "../data/wikidata_tekgen/test/ont_1_movie_ground_truth.jsonl"
    
    )


    """
    def __init__(self, ontology_path: str, sent_file_path): 
        self.ontology_path = ontology_path
        self.sent_file_path = sent_file_path

    def getPromptOf(test_sentence: dict, n_examples: int) -> str:
        pass


"""
{
    "id": "ont_1_movie_test_1", 
    "sent": "Bleach: Hell Verse (Japanese: BLEACH , Hepburn: Bur\u00c4\u00abchi Jigoku-Hen) is a 2010 Japanese animated film directed by Noriyuki Abe.", 
    "triples": [
        {"sub": "Bleach : Hell Verse", "rel": "director", "obj": "Noriyuki Abe"}, 
        {"sub": "Bleach : Hell Verse", "rel": "publication date", "obj": "01 January 2010"}
    ], 
    "unseen": false, 
    "verified": true, 
    "similars": [
        "ont_1_movie_train_27", 
        "ont_1_movie_train_612", 
        "ont_1_movie_train_715", 
        "ont_1_movie_train_67", 
        "ont_1_movie_train_119"
    ]
}
"""

