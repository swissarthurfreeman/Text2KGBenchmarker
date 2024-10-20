from typing import List
import pprint as pp
import json

class Prompter:
    """
    Allows generating prompts from sentences with a description of the 
    ontolgoy and a specified number of examples. Relation labels have 
    their spaces replaced by '_', e.g. 'publication date' will get 
    mapped to 'publication_date'.
    
    Usage
    -----
    >>> prompter = Prompter(
            "../data/wikidata_tekgen/ontologies/1_movie_ontology.json"
            "../data/wikidata_tekgen/train/ont_1_movie_train.jsonl"
            "../data/wikidata_tekgen/test/ont_1_movie_test.jsonl"
        )

    >>> prompter.getPromptOf("ont_1_movie_test_53", n_examples=2)
    ```
    Given the following ontology and sentences, please extract the triples from the
    sentence  according to the relations in the ontology. In the output, only include
    the triples in the given output format.
    
    CONTEXT:

    Ontology Concepts: 
    human, city, country, film, film genre, genre, film production company, 
    film award, award,  written work, film character, film organization
    
    Ontology Relations: 
    director(film,human), screenwriter(film,human), genre(film,genre), 
    based_on(film,written work), cast_member(film,human), award_received(film,award), 
    production_company(film,film production company), country_of_origin(film,country), 
    publication_date(film,), characters(film,film character), 
    narrative_location(film,city), filming_location(film,city), 
    main_subject(film,), nominated_for(film,award), cost(film,)
    
    Example Sentence: 
    Knighty Knight Bugs is a 1958 Warner Bros. Looney Tunes cartoon directed 
    by Friz Freleng, The short was released on August 23, 1958, and stars Bugs Bunny.

    Example Output:
    director(Knighty Knight Bugs,Friz Freleng)

    Example Sentence:
    The Prize Pest is a 1951 Warner Bros. Looney Tunes cartoon directed by 
    Robert McKimson, and written by Tedd Pierce.
    
    Example Output:
    screenwriter(The Prize Pest,Tedd Pierce)

    Test Sentence: 
    Yankee Doodle Bugs is a 1954 Warner Bros. Looney Tunes cartoon short, 
    written by Warren Foster and directed by Friz Freleng.

    Test Output:
    ```
    

    """
    def __init__(self, ontology_path: str, train_sent_path: str, test_sent_path: str): 
        self.ontology_description: str = self._getOntologyDescription(ontology_path)
        self.train_sent_path = train_sent_path
        self.train_sentences = self.load_jsonl_as_dict(self.train_sent_path)
        self.test_sent_path = test_sent_path
        self.test_sentences = self.load_jsonl_as_dict(self.test_sent_path)

    def load_jsonl_as_dict(self, path: str) -> dict[str, dict[str, bool|list[str]|list[dict[str, str]]]]:
        with open(path, "r") as f:
            sent_dicts = [json.loads(line) for line in f]
            
            res = {}
            for sent_dict in sent_dicts:
                res[sent_dict['id']] = {}
                for key in sent_dict:
                    res[sent_dict['id']][key] = sent_dict[key]
                    
            return res
                    

    def getPromptOf(self, test_sentence_id: str, n_examples: int) -> str:
        """
        
        Parameters
        ----------
        - test_sentence_id: dict
        - n_examples: int
        """
        res = self.getSystemInstructions() + "\n\nCONTEXT:\n\n"
        res += self.ontology_description
        test_sentence = self.test_sentences[test_sentence_id]
        
        # TODO : if n_examples is larger than len(test_sentence[similars]) we should sample non similar sentence from train data.
        for idx in range(len(test_sentence["similars"][:n_examples])):
            train_sent_id = test_sentence["similars"][idx]
            train_sentence, train_triples = self.train_sentences[train_sent_id]["sent"], self.train_sentences[train_sent_id]["triples"]
            res += "\n\nExample Sentence : " +  train_sentence + "\n\nExample Output:\n"
            for triple in train_triples:
                res += triple["rel"].replace(" ", "_") + "(" + triple["sub"] + ", " + triple["obj"] + ")" + "\n"
        
        res += "\n\nTest Sentence: " + test_sentence["sent"] + "\n\nTest Output: "
        return res

    def getAllTestPrompts(n_examples: int) -> dict[str, str]:
        """Get dict {"sent_id": "prompt"} of all prompts of test sentences of dataset with `n_examples` train examples."""
        pass

    def _getOntologyDescription(self, ontology_path) -> str:
        res = ""
        with open(ontology_path, "r") as onto_f:
            onto = json.load(onto_f)
            res = "Ontology Concepts:\n"
            res += self.getConceptsOf(onto["concepts"])
            res += "\n\nOntology Relations:\n"
            res += self.getRelationsOf(onto["relations"])
        return res
        
    def getConceptsOf(self, concepts: List[dict]) -> str:
        res = ""
        for concept in concepts:
            res += concept["label"] + ", "
        return res[:-2]     # ignore extra comma
    
    
    def getRelationsOf(self, relations: List[dict]) -> str:
        res = ""
        for relation in relations:
            res += relation["label"].replace(" ", "_") + "(" + relation["domain"] + ", " + relation["range"] + "), "
        return res[:-2]     # ignore extra comma

    def getSystemInstructions(self) -> str:
        return """Given the following ontology and sentences, please extract the triples from the sentence according to the relations in the ontology. \nIn the output, only include the triples in the given output format."""


if __name__ == "__main__":
    wikidata_prompter = Prompter(
        "../../data/wikidata_tekgen/ontologies/1_movie_ontology.json",
        "../../data/wikidata_tekgen/train/ont_1_movie_train.jsonl",
        "../../data/wikidata_tekgen/test/ont_1_movie_test.jsonl"
    )
    
    #print(wikidata_prompter.getPromptOf("ont_1_movie_test_1", n_examples=3))

    dpedia_webnlg_prompter = Prompter(
        "../../data/dpedia_webnlg/ontologies/6_politician_ontology.json",
        "../../data/dpedia_webnlg/train/ont_6_politician_train.jsonl",
        "../../data/dpedia_webnlg/test/ont_6_politician_test.jsonl"
    )
    print(dpedia_webnlg_prompter.getPromptOf("ont_6_politician_test_1", n_examples=3))


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

