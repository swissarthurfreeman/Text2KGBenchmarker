from typing import List
import pprint as pp
import random
import json

def load_jsonl_as_dict(path: str) -> dict[str, dict[str, bool|list[str]|list[dict[str, str]]]]:
    with open(path, "r") as f:
        sent_dicts = [json.loads(line) for line in f]
        
        res = {}
        for sent_dict in sent_dicts:
            res[sent_dict['id']] = {}
            for key in sent_dict:
                res[sent_dict['id']][key] = sent_dict[key]
                
        return res

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
        self.train_sentences = load_jsonl_as_dict(self.train_sent_path)
        self.test_sent_path = test_sent_path
        self.test_sentences = load_jsonl_as_dict(self.test_sent_path)
                    

    def getPromptOf(self, test_sentence_id: str, n_examples: int) -> str:
        """
        
        Parameters
        ----------
        - test_sentence_id: dict
        - n_examples: int
        """
        res = self.getSystemInstructions() + "\n\nCONTEXT:\n\n"
        res += self.ontology_description
        try:
            test_sentence = self.test_sentences[test_sentence_id]
        except KeyError:
            print("Test sentence id " + test_sentence_id + " doesn't exist in " + self.test_sent_path + " file.")
            exit(1)
        
        # TODO : if n_examples is larger than len(test_sentence[similars]) we should sample non similar sentence from train data.
        similar_train_sent_ids: list[str] = test_sentence["similars"][:n_examples]
        if n_examples > len(test_sentence["similars"]):
            print("WARNING: " + test_sentence["id"] + " does not have ", n_examples, " similars, sampling missing examples from train data instead")
            for _ in range(n_examples - len(test_sentence["similars"])):
                # append a bunch of random sentences
                random_train_sent_id = list(self.train_sentences.keys())[random.randint(0, len(self.train_sentences.keys()))]
                similar_train_sent_ids.append(self.train_sentences[random_train_sent_id]["id"])
        
        for similar_train_sent_id in similar_train_sent_ids:
            train_sentence, train_triples = self.train_sentences[similar_train_sent_id]["sent"], self.train_sentences[similar_train_sent_id]["triples"]
            res += "\n\nExample Sentence : " +  train_sentence + "\n\nExample Output:\n"
            for triple in train_triples:
                res += triple["rel"].strip().replace(" ", "_") + "(" + triple["sub"].strip() + ", " + triple["obj"].strip() + ")" + "\n"
        
        res += "\n\nTest Sentence: " + test_sentence["sent"] + "\n\nTest Output: "
        return res

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
            res += concept["label"].strip() + ", "
        return res[:-2]     # ignore extra comma
    
    
    def getRelationsOf(self, relations: List[dict]) -> str:
        res = ""
        for relation in relations:
            res += relation["label"].strip().replace(" ", "_") + "(" + relation["domain"] + ", " + relation["range"] + "), "
        return res[:-2]     # ignore extra comma

    def getSystemInstructions(self) -> str:
        return """Given the following ontology and sentences, please extract the triples from the sentence according to the relations in the ontology. \nIn the output, only include the triples in the given output format, if you can't extract triples, leave the output empty. Do not include any formatting backticks like ``` or any notes or remarks. Extract as many triples as possible."""


if __name__ == "__main__":
    wikidata_prompter = Prompter(
        "../../data/wikidata_tekgen/ontologies/ont_9_nature.json",
        "../../data/wikidata_tekgen/train/ont_9_nature_train.jsonl",
        "../../data/wikidata_tekgen/test/ont_9_nature_test.jsonl"
    )
    
    print(wikidata_prompter.getPromptOf("ont_9_nature_unseen_test_19", n_examples=10))
    
    """
    dpedia_webnlg_prompter = Prompter(
        "../../data/dpedia_webnlg/ontologies/ont_6_politician.json",
        "../../data/dpedia_webnlg/train/ont_6_politician_train.jsonl",
        "../../data/dpedia_webnlg/test/ont_6_politician_test.jsonl"
    )
    """
    
    #print(dpedia_webnlg_prompter.getPromptOf("ont_6_politician_test_1", n_examples=3))
