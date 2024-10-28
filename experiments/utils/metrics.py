import os
import glob
import json
import re
from prompter import load_jsonl_as_dict
from nltk.tokenize import word_tokenize
from nltk.stem import PorterStemmer
import nltk
nltk.download('punkt_tab')

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

class LLMMetrics():
    """
    Helper class to compute all metrics for a single run like wikidata_tekgen train / test n_examples 2 on movie ontology.
    """
    def __init__(self, onto_path: str, test_sent_path: str):
        
        
        self.test_sent_path = test_sent_path
        self.test_sentences: dict = load_jsonl_as_dict(self.test_sent_path)
        
        with open(onto_path, "r") as onto_f:
            concepts = []
            relations = []
            ontology = json.load(onto_f)
            
            for concept in ontology["concepts"]:
                concepts.append(concept["label"])
            
            for rel in ontology["relations"]:
                relations.append(rel["label"])
            self.onto_concepts: list[str] = concepts
            self.onto_relations: list[str] = relations
        
    def computeMetricsPerReponseOf(self, llm_response_file_path: str) -> None:
        """Open `llm_response_file` read all sentences, compute metrics for every sentence,
        add average over all sentences of the ontology to the averages file."""
        model_name = llm_response_file_path.split("/")[-2]
        test_data_name = llm_response_file_path.split("/")[-1].split("-")[1]
        train_data_name = llm_response_file_path.split("/")[-1].split("-")[2]
        n_examples = int(llm_response_file_path.split("/")[-1].split("_")[-1][0])
        ontology_name = llm_response_file_path.split("/")[-1].split("-")[0]
        
        with open(llm_response_file_path) as llm_response_f:
            llm_responses: list[dict] = [json.loads(line) for line in llm_response_f]
            for response in llm_responses:
                r: dict = self.addMetricsToResponse(response)
                if not os.path.exists("../results/metrics/" + model_name):
                    os.mkdir("../results/metrics/" + model_name)
                
                with open("../results/metrics/" + model_name + "/" + llm_response_file_path.split("/")[-1][:-6] + "_metrics.jsonl", "a") as out_f:
                    out_f.write(json.dumps(r) + "\n")
                
                self.test_sentences[r["id"]]["precision"] = r["precision"]        # keep track of metrics in self
                self.test_sentences[r["id"]]["recall"] = r["recall"]
                self.test_sentences[r["id"]]["f1"] = r["f1"]
                self.test_sentences[r["id"]]["onto_conf"] = r["onto_conf"]
                self.test_sentences[r["id"]]["sub_halluc"] = r["sub_halluc"]
                self.test_sentences[r["id"]]["rel_halluc"] = r["rel_halluc"]
                self.test_sentences[r["id"]]["obj_halluc"] = r["obj_halluc"]
        
        average_metrics_path = "../results/metrics/" + model_name + "/" + "-".join([test_data_name, train_data_name]) + "-n_examples_" + str(n_examples) + "_averages.jsonl"
        self.computeAverageMetrics(ontology_name, average_metrics_path)
    
    def computeAverageMetrics(self, ontology_name: str, average_metrics_path: str) -> None:
        metrics = {
            "onto": ontology_name,
            "unseen": {"n_sentences": self.get_nUnseen(), "avg_precision": 0, "avg_recall": 0, "avg_f1": 0, "avg_onto_conf": 0, "avg_sub_halluc": 0, "avg_rel_halluc": 0, "avg_obj_halluc": 0}, 
            "verified": {"n_sentences": self.get_nVerified(), "avg_precision": 0, "avg_recall": 0, "avg_f1": 0, "avg_onto_conf": 0, "avg_sub_halluc": 0, "avg_rel_halluc": 0, "avg_obj_halluc": 0}, 
            "all": {"n_sentences": len(self.test_sentences.keys()), "avg_precision": 0, "avg_recall": 0, "avg_f1": 0, "avg_onto_conf": 0, "avg_sub_halluc": 0, "avg_rel_halluc": 0, "avg_obj_halluc": 0}
        }
        
        def _average(test_sent_w_metrics, metrics, sent_type):
            metrics[sent_type]["avg_precision"] += test_sent_w_metrics["precision"] / metrics[sent_type]["n_sentences"]
            metrics[sent_type]["avg_recall"] += test_sent_w_metrics["recall"] / metrics[sent_type]["n_sentences"]
            metrics[sent_type]["avg_f1"] += test_sent_w_metrics["f1"] / metrics[sent_type]["n_sentences"]
            
            metrics[sent_type]["avg_onto_conf"] += test_sent_w_metrics["onto_conf"] / metrics[sent_type]["n_sentences"] 
            metrics[sent_type]["avg_sub_halluc"] += test_sent_w_metrics["sub_halluc"] / metrics[sent_type]["n_sentences"]
            metrics[sent_type]["avg_rel_halluc"] += test_sent_w_metrics["rel_halluc"] / metrics[sent_type]["n_sentences"] 
            metrics[sent_type]["avg_obj_halluc"] += test_sent_w_metrics["obj_halluc"] / metrics[sent_type]["n_sentences"]

        
        for test_sent_w_metrics in self.test_sentences.values():
            if ("unseen" in test_sent_w_metrics.keys()) and test_sent_w_metrics["unseen"]: _average(test_sent_w_metrics, metrics, "unseen")
            if ("verified" in test_sent_w_metrics.keys()) and test_sent_w_metrics["verified"]: _average(test_sent_w_metrics, metrics, "verified")
            _average(test_sent_w_metrics, metrics, "all")
                
        with open(average_metrics_path, "a") as avg_f:
            avg_f.write(json.dumps(metrics) + "\n")
            
    def get_nUnseen(self) -> int:
        res = 0
        for test_sent in self.test_sentences.values():
            if ("unseen" in test_sent) and test_sent["unseen"]:
                res += 1
        return res
    
    def get_nVerified(self) -> int:
        res = 0
        for test_sent in self.test_sentences.values():
            if ("verified" in test_sent) and test_sent["verified"]:
                res += 1
        return res
    
    def normalize_triple(self, sub: str, rel: str, obj: str) -> str:
        # remove spaces and underscores and make lower case
        sub_n = re.sub(r"(_|\s+)", '', sub).lower()
        rel_n = re.sub(r"(_|\s+)", '', rel).lower()
        obj_n = re.sub(r"(_|\s+)", '', obj).lower()
        # concatenate them to a single string
        return f"{sub_n}{rel_n}{obj_n}"
    
    def makeNormalizedSet(self, triples: list[dict]) -> set[str]:
        res = set()
        for triple in triples:
            triple_string = self.normalize_triple(triple["sub"], triple["rel"], triple["obj"])
            res.add(triple_string)
        return res
    
    def computePRF1(self, response: dict) -> tuple[float, float, float]:
        ground_truth: set[str] = self.makeNormalizedSet(self.test_sentences[response["sent_id"]]["triples"])
        llm_triples: set[str] = self.makeNormalizedSet(response["triples"])

        print(ground_truth)
        print(llm_triples)
        if len(llm_triples) == 0 or len(ground_truth) == 0: return 0, 0, 0
        
        P = len(ground_truth.intersection(llm_triples)) / len(llm_triples)
        R = len(ground_truth.intersection(llm_triples)) / len(ground_truth)
        
        if P + R > 0:
            F1 = 2 *((P*R) / (P + R))
        else:
            F1 = 0
        return P, R, F1
    
    def addMetricsToResponse(self, response: dict) -> dict:
        P, R, F1, OC, SH, RH, OH = self.computeMetricsOf(response)
        r = {}
        r["id"] = response["sent_id"]
        r["llm_triples"] = response["triples"]
        r["precision"] = P
        r["recall"] = R
        r["f1"] = F1
        r["onto_conf"] = OC
        r["rel_halluc"] = RH
        r["sub_halluc"] = SH
        r["obj_halluc"] = OH
        return r
    
    def computeOCSHRHOH(self, response: dict) -> tuple[float, float, float, float]:
        OC, RH = self.get_ontology_conformance(response)
        SH, OH = self.get_subject_object_hallucinations(response) 
        return OC, SH, RH, OH
    
    def computeMetricsOf(self, response: dict) -> tuple[float, float, float, float, float, float, float]:
        """Return P, R, F1, OC, SH, RH, OH"""
        P, R, F1 = self.computePRF1(response)
        OC, SH, RH, OH = self.computeOCSHRHOH(response)
        return P, R, F1, OC, SH, RH, OH
    
    def get_subject_object_hallucinations(self, response: dict) -> tuple[float, float]:
        """
        Calculate subject and object hallucinations metrics. As the context for calculating hallucinations, we consider the
        test sentence and the ontology concepts as relevant tokens.
        :param ps: stemmer for stemming words before checking for hallucinations
        :param ontology: ontology to take into account with the concepts and relations
        :param test_sentence: test sentences for which the triples are generated
        :param triples: a set of triples generated by the system
        :return:
            subj_hallucination: float - subject hallucination metric
            obj_hallucination: float - object hallucination metric
        """
        ps = PorterStemmer()

        # if the set of triples are empty, we return 0
        if len(response["triples"]) == 0:
            return 0, 0

        # append the test sentence with concepts from the ontology
        test_sentence = self.test_sentences[response["sent_id"]]["sent"]
        test_sentence += " ".join(self.onto_concepts)
        # stem each word in the test sentence concatenated with the ontology concepts
        stemmed_sentence = "".join([ps.stem(word) for word in word_tokenize(test_sentence)])
        # normalize the text to remove white spaces and underscores
        normalized_stemmed_sentence = re.sub(r"(_|\s+)", '', stemmed_sentence).lower()

        # count the number of subject and object hallucinations
        num_subj_hallucinations, num_obj_hallucinations = 0, 0
        for triple in response["triples"]:
            # clean and normalize subject and object noun phrases the same way as the test sentence
            normalized_stemmed_subject = self.clean_entity_string(ps, triple["sub"])
            normalized_stemmed_object = self.clean_entity_string(ps, triple["obj"])

            # check if the subject/object is found in the stemmed sentence/context text. If not found, mark it as a hallucination
            if normalized_stemmed_sentence.find(normalized_stemmed_subject) == -1:
                num_subj_hallucinations += 1
            if normalized_stemmed_sentence.find(normalized_stemmed_object) == -1:
                num_obj_hallucinations += 1

        # divide the number of hallucinations by the number of triples to calculate the hallucination metrics
        subj_hallucination = num_subj_hallucinations / len(response["triples"])
        obj_hallucination = num_obj_hallucinations / len(response["triples"])
        return subj_hallucination, obj_hallucination

    def clean_entity_string(self, ps, entity: str) -> str:
        """
        Utility method to clean subject and object strings of triples
        :param ps: stemmer for stemming words before checking for hallucinations
        :param entity: subject or object string
        :return: the cleaned and normalized string
        """
        # stem every word for better matches
        stemmed_entity = "".join([ps.stem(word) for word in word_tokenize(entity)])
        # normalizing the string by removing white spaces, underscores and then converting to lower case
        normalized_stemmed_entity = re.sub(r"(_|\s+)", '', stemmed_entity).lower()
        # special handling for string with years to remove January 01
        return normalized_stemmed_entity.replace("01januari", "")

    def get_ontology_conformance(self, response: dict) -> tuple[float, float]:
        """
        Calculate the ontology conformance and relation hallucination metrics.
        :param ontology: ontology to take into account with the concepts and relations
        :param triples: a set of triples generated by the system
        :return:
            ont_conformance: float - ontology conformance metric
            rel_hallucination: float - relation hallucination metric = 1 - ontology conformance
        """
        if len(response["triples"]) == 0:
            return 1, 0
        # replace spaces with underscores in the ontology relations
        ont_rels = [rel.replace(" ", "_") for rel in self.onto_relations]
        # count the number of system triples relations that are in the ontology
        num_rels_conformant = len([tr for tr in response["triples"] if tr["rel"] in ont_rels])

        # ontology conformance is the number of system triples relations in the ontology divided by the total number of system triples
        ont_conformance = num_rels_conformant / len(response["triples"])
        # relation hallucination is 1 - ontology conformance
        rel_hallucination = 1 - ont_conformance
        return ont_conformance, rel_hallucination
    
    
    
if __name__ ==  "__main__": 
    for ontology_name in dpedia_webnlg_files:
        
        l = LLMMetrics(
            "../../data/dpedia_webnlg/ontologies/" + ontology_name + ".json",
            "../../data/dpedia_webnlg/test/" + ontology_name + "_test.jsonl"
        )
        
        files = glob.glob("../results/llm_responses/Babelscape.rebel-large/" + ontology_name + "-*")
        for llm_response_files_for_ontology_n_examples in files:
            l.computeMetricsPerReponseOf(llm_response_files_for_ontology_n_examples)
    
    
    for ontology_name in wikidata_tekgen_files:
        l = LLMMetrics(
            "../../data/wikidata_tekgen/ontologies/" + ontology_name + ".json",
            "../../data/wikidata_tekgen/test/" + ontology_name + "_test.jsonl"
        )
        
        files = glob.glob("../results/llm_responses/Babelscape.rebel-large/" + ontology_name + "-*")
        for llm_response_files_for_ontology_n_examples in files:
            l.computeMetricsPerReponseOf(llm_response_files_for_ontology_n_examples)
    
    """
    def computeAllMetrics(self) -> None:
        for n_examples in [1, 2, 3, 4, 5, 6]:
            llm_response_file_paths = glob.glob(self.llm_responses_folder_path + "/*n_examples_" + str(n_examples) + ".jsonl")
        
            for llm_response_file_path in llm_response_file_paths:
                self.computeMetricsPerReponseOf(llm_response_file_path)
    """