import json
import random
from utils import getOntologyConceptsList, getOntologyRelationsList, load_jsonl_as_dict


class Prompter:
    """
    Allows generating prompts from sentences with a description of the 
    ontolgoy and a specified number of examples. Relation labels have 
    their spaces replaced by '_', e.g. 'publication date' will get 
    mapped to 'publication_date'.
    
    Usage
    -----
    >>> wikidata_prompter = Prompter("wikidata_tekgen", "ont_9_nature")

    >>> wikidata_prompter.getPromptOf("ont_1_movie_test_53", n_examples=2)
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
    def __init__(self, dataset_name: str, ontology_name: str): 
        self._dataset_name, self._ontology_name = dataset_name, ontology_name
        self.ontology_description: str = self._getOntologyDescription()
        self.train_sent_path = "../../data/" + dataset_name + "/train/" + ontology_name + "_train.jsonl"
        self.train_sentences = load_jsonl_as_dict(self.train_sent_path)
        self.test_sent_path = "../../data/" + dataset_name + "/test/" + ontology_name + "_test.jsonl"
        self.test_sentences = load_jsonl_as_dict(self.test_sent_path)
                    

    def getPromptOf(self, test_sentence_id: str, n_examples: int) -> str:
        """
        Get prompt for LLM, explains the ontology concepts and relation as well as the triple 
        extraction task with `n_examples` examples. 
        
        Parameters
        ----------
        - test_sentence_id: dict
        - n_examples: int
        """
        res = self._getSystemInstructions() + "\n\nCONTEXT:\n\n"
        res += self.ontology_description
        try:
            test_sentence = self.test_sentences[test_sentence_id]
        except KeyError:
            print("Test sentence id " + test_sentence_id + " doesn't exist in " + self.test_sent_path + " file.")
            exit(1)
        
        similar_train_sent_ids: list[str] = test_sentence["similars"][:n_examples]
        if n_examples > len(test_sentence["similars"]):
            # if n_examples is larger than len(test_sentence[similars]) we sample non similar sentence from train data.
            print("WARNING: " + test_sentence["id"] + " does not have ", n_examples, " similars, sampling missing examples from train data instead")
            for _ in range(n_examples - len(test_sentence["similars"])):
                # append a bunch of random sentences
                random_train_sent_id = list(self.train_sentences.keys())[random.randint(0, len(self.train_sentences.keys())-1)]
                similar_train_sent_ids.append(self.train_sentences[random_train_sent_id]["id"])
        
        for similar_train_sent_id in similar_train_sent_ids:
            train_sentence, train_triples = self.train_sentences[similar_train_sent_id]["sent"], self.train_sentences[similar_train_sent_id]["triples"]
            res += "\n\nExample Sentence : " +  train_sentence + "\n\nExample Output:\n"
            for triple in train_triples:
                res += triple["rel"].strip().replace(" ", "_") + "(" + triple["sub"].strip() + ", " + triple["obj"].strip() + ")" + "\n"
        
        res += "\n\nTest Sentence: " + test_sentence["sent"] + "\n\nTest Output: "
        return res

    def _getOntologyDescription(self) -> str:
        res = ""
        res = "Ontology Concepts:\n"
        res += ", ".join(getOntologyConceptsList(self._ontology_name, self._dataset_name))
        res += "\n\nOntology Relations:\n"
        res += ", ".join(getOntologyRelationsList(self._ontology_name, self._dataset_name))
        return res
        
    def _getSystemInstructions(self) -> str:
        return """Given the following ontology and sentences, please extract the triples from the sentence according to the relations in the ontology. \nIn the output, only include the triples in the given output format, if you can't extract triples, leave the output empty. Do not include any formatting backticks like ``` or any notes or remarks. Extract as many triples as possible."""


if __name__ == "__main__":
    #wikidata_prompter = Prompter("wikidata_tekgen", "ont_9_nature")
    #print(wikidata_prompter.getPromptOf("ont_9_nature_unseen_test_19", n_examples=5))
    
    
    dpedia_prompter = Prompter("dpedia_webnlg", "ont_4_building")
    print(dpedia_prompter.getPromptOf("ont_4_building_test_1", n_examples=5))
    
    