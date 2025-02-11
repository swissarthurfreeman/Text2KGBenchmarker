import re
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
    >>> ont = "ont_1_movie"
    >>> wikidata_prompter = Prompter("wikidata_tekgen", f"{ont}")
    >>> print(wikidata_prompter.getPromptOf(f"{ont}_test_20", n_examples=4))
    ```
    WARNING: ont_1_movie_test_20 does not have  4  similars, sampling missing examples from train data instead
    Given the following ontology and sentences, please extract the triples from the sentence according to the relations in the ontology. 
    In the output, only include the triples in the given output format, if you can't extract triples, leave the output empty. 
    Do not include any formatting backticks like ``` or any notes or remarks. Extract as many triples as possible.

    CONTEXT:

    Ontology Concepts:
    human, city, country, film, film genre, genre, film production company, film award, award, written work, film character, film organization

    Ontology Relations:
    director(film | human), screenwriter(film | human), genre(film | genre), based_on(film | written work), cast_member(film | human), 
    award_received(film | award), production_company(film | film production company), country_of_origin(film | country), 
    publication_date(film | literal), characters(film | film character), narrative_location(film | city), filming_location(film | city), 
    main_subject(film | literal), nominated_for(film | award), cost(film | literal)

    Example Sentence : The film also features the return of Adrienne King, Betsy Palmer and Walt Gorney, who respectively portrayed Alice Hardy, 
    Pamela Voorhees, and Crazy Ralph in the prior installment.

    Example Output:
    characters(Friday the 13th Part 2 | Pamela Voorhees)


    Example Sentence : The film stars Roy Scheider (replacing William Sylvester), Helen Mirren, Bob Balaban and John Lithgow, 
    along with Keir Dullea and Douglas Rain of the cast of the previous film.

    Example Output:
    cast_member(2010: The Year We Make Contact | Douglas Rain)
    cast_member(2010: The Year We Make Contact | Bob Balaban)
    cast_member(2010: The Year We Make Contact | John Lithgow)


    Example Sentence : Mouna Ragam was the first film produced by Venkateswaran's Sujatha Films, and was shot primarily in Madras, 
    with additional filming taking place in Delhi and Agra.

    Example Output:
    filming_location(Mouna Ragam | Delhi)


    Example Sentence : Set in the late 19th century, the novel recounts the adventures of Anne Shirley, an 11-year-old orphan girl, 
    who is mistakenly sent to two middle-aged siblings, Matthew and Marilla Cuthbert, who had originally intended to adopt a boy to 
    help them on Anne of Green Gables's farm in the fictional town of Avonlea on Prince Edward Island, Canada.

    Example Output:
    country_of_origin(Anne of Green Gables | Canada)


    Test Sentence: The film is directed by Raja Gosnell, who helmed the first, with all the main cast returning.

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
                res += triple["rel"].strip().replace(" ", "_") + "("
                res += triple["sub"].strip() + " | "    
                res += triple["obj"].strip() + ")" + "\n"
                    
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
        return """Given the following ontology and sentences, please extract the triples from the sentence according to the relations in the ontology. \n Makes sure to respect domain and range constraints. \nIn the output, only include the triples in the given output format, if you can't extract triples, leave the output empty. Do not include any formatting backticks like ``` or any notes or remarks. Extract as many triples as possible."""


if __name__ == "__main__":
    ont = "ont_2_musicalwork"

    dbpedia_prompter = Prompter("dbpedia_webnlg_clean", ont)
    print(dbpedia_prompter.getPromptOf(f"{ont}_test_5", n_examples=2))
    
    #wikidata_prompter = Prompter("wikidata_tekgen", f"{ont}")
    #print(wikidata_prompter.getPromptOf(f"{ont}_test_102", n_examples=3))
    