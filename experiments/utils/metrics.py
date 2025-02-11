import os
from pprint import pprint
import re
import glob
import json
import nltk
import numpy as np
nltk.download('punkt_tab')
from nltk.stem import PorterStemmer
from nltk.tokenize import word_tokenize
from utils import DBPEDIA_WEBNLG_ONT_NAMES, WIKIDATA_TEKGEN_ONT_NAMES
from utils import load_jsonl_as_dict, getOntologyConceptsList, getOntologyRelationsListLabels, camelCaseToSpaces, load_jsonl_as_list


class LLMMetrics():
    """
    Helper class to compute all metrics for a single ontology of a given dataset.
    An instance should only be used once via it's `generate()` method and
    then discarded. 
    """
    def __init__(self, llm_responses_folder_name: str, ontology_name: str, dataset_name: str):
        self.llm_responses_folder_name, self.ontology_name, self.dataset_name = llm_responses_folder_name, ontology_name, dataset_name
        self.test_sent_path = "../../data/" + dataset_name + "/test/" + ontology_name + "_test.jsonl" 
        self.test_sentences: dict = load_jsonl_as_dict(self.test_sent_path)
        
        self.n_unseen, self.n_verified = 0, 0
        for test_sent in self.test_sentences.values():
            if ("unseen" in test_sent) and test_sent["unseen"]: self.n_unseen += 1
            if ("verified" in test_sent) and test_sent["verified"]: self.n_verified += 1
        
        self.onto_concepts: list[str] = getOntologyConceptsList(ontology_name, dataset_name)
        """list of raw ontology concepts, surface form words seperated by spaces, no camelcase."""
        self.onto_relations: list[str] = getOntologyRelationsListLabels(ontology_name, dataset_name)
        """list of ontology relations, surface form words seperated by spaces."""
        
        # if few shot setting, llm_results will have one folder per n_shot, e.g. gpt-4o-n_examples=4, considered as a seperate technique.
        # TODO : here we replace "_clean" with "", this should only be used for REBEL, or anywhere ontology is not provided to the model
        # this is for the sake of computing performance of REBEL with cleaned webnlg dataset instead of camelcase/underscore one.
        self.llm_responses_path = "../results/llm_responses/" + self.llm_responses_folder_name + "/" + self.ontology_name + "-" + self.dataset_name + ".jsonl"
        self.metrics_dir = "../results/metrics/" + llm_responses_folder_name
        self.avg_met_path = self.metrics_dir + "/" + self.dataset_name + "_avg.jsonl"
        if not os.path.exists(self.metrics_dir): os.makedirs(self.metrics_dir)
        
        self.avg_met = {
            "onto": self.ontology_name,
            "unseen": {"n_sentences": self.n_unseen, "avg_precision": 0, "avg_recall": 0, "avg_f1": 0, "avg_onto_conf": 0, "avg_sub_halluc": 0, "avg_rel_halluc": 0, "avg_obj_halluc": 0}, 
            "verified": {"n_sentences": self.n_verified, "avg_precision": 0, "avg_recall": 0, "avg_f1": 0, "avg_onto_conf": 0, "avg_sub_halluc": 0, "avg_rel_halluc": 0, "avg_obj_halluc": 0}, 
            "all": {"n_sentences": len(self.test_sentences.keys()), "avg_precision": 0, "avg_recall": 0, "avg_f1": 0, "avg_onto_conf": 0, "avg_sub_halluc": 0, "avg_rel_halluc": 0, "avg_obj_halluc": 0}
        }
        """Average metrics for the specified llm_responses, ontology and dataset."""
        self.ps = PorterStemmer()
                    
    def generate(self) -> None:
        """Compute performance metrics for specified `llm_responses_folder_name`, `ontology_name` and `dataset_name`, store
        metrics within `../results/metrics/llm_responses_folder_name/ontology_name-dataset_name.jsonl`, add average metrics 
        line to `../results/metrics/llm_responses_folder/dataset_name_avg.jsonl`"""
    
        llm_responses: list[dict] = load_jsonl_as_list(self.llm_responses_path)
        for response in llm_responses:
            response: dict = self._add_metrics_to(response)
            
            with open(self.metrics_dir + "/" + self.ontology_name + "-" + self.dataset_name + ".jsonl", "a") as out_f:
                out_f.write(json.dumps(response) + "\n")
            
            self._add_to_average(response)
    
        with open(self.avg_met_path, "a") as avg_f:
            avg_f.write(json.dumps(self.avg_met) + "\n")
    
    def _add_to_average(self, test_sent_w_metrics):
        types = ["all"]
        if "unseen" in self.test_sentences[test_sent_w_metrics['id']].keys() and self.test_sentences[test_sent_w_metrics['id']]["unseen"]: types.append("unseen")        # if the sentence is unseen or verified
        if "verified" in self.test_sentences[test_sent_w_metrics['id']].keys() and self.test_sentences[test_sent_w_metrics['id']]["verified"]: types.append("verified")    # we'll add to the averages of that slice
        for typ in types:
            if(self.avg_met[typ]["n_sentences"] == 0): continue
            self.avg_met[typ]["avg_precision"] += test_sent_w_metrics["precision"] / self.avg_met[typ]["n_sentences"]
            self.avg_met[typ]["avg_recall"] += test_sent_w_metrics["recall"] / self.avg_met[typ]["n_sentences"]
            self.avg_met[typ]["avg_f1"] += test_sent_w_metrics["f1"] / self.avg_met[typ]["n_sentences"]
            
            self.avg_met[typ]["avg_onto_conf"] += test_sent_w_metrics["onto_conf"] / self.avg_met[typ]["n_sentences"] 
            self.avg_met[typ]["avg_sub_halluc"] += test_sent_w_metrics["sub_halluc"] / self.avg_met[typ]["n_sentences"]
            self.avg_met[typ]["avg_rel_halluc"] += test_sent_w_metrics["rel_halluc"] / self.avg_met[typ]["n_sentences"] 
            self.avg_met[typ]["avg_obj_halluc"] += test_sent_w_metrics["obj_halluc"] / self.avg_met[typ]["n_sentences"]            
    
    def _add_metrics_to(self, res: dict) -> dict:
        """Add performance metrics as keys to response dict, P, R, F1, OC, SH, RH, OH."""
        res["precision"], res["recall"], res["f1"] = self._compute_prec(res)
        res["onto_conf"], res["rel_halluc"] = self._get_ontology_conformance_RH(res) 
        res["sub_halluc"], res["obj_halluc"] = self._get_subject_object_hallucinations(res)
        return res
    
    def _compute_prec(self, response: dict) -> tuple[float, float, float]:
        """Compute and return classical precision measures, Precision, Recall, F1."""
        ground_truth: set[str] = self._get_normalized_triple_set(self.test_sentences[response["id"]]["triples"])
        llm_triples: set[str] = self._get_normalized_triple_set(response["triples"])

        if len(llm_triples) == 0 or len(ground_truth) == 0: return 0, 0, 0
        
        P = len(ground_truth.intersection(llm_triples)) / len(llm_triples)
        R = len(ground_truth.intersection(llm_triples)) / len(ground_truth)
        
        if P + R > 0: F1 = 2 *((P*R) / (P + R)) 
        else: F1 = 0
        return P, R, F1
    
    def _get_normalized_triple_set(self, triples: list[dict]) -> set[str]:
        """Return clean set of triples, where every triple is cleaned via `_normalize_triple`"""
        res = set()
        for triple in triples:
            triple_string = self._normalize_triple(triple["sub"], triple["rel"], triple["obj"])
            res.add(triple_string)
        return res
    
    def _normalize_triple(self, sub: str, rel: str, obj: str) -> str:
        """Remove spaces/underscores and make lower case sub, rel, obj, return a single string."""
        sub_n = re.sub(r"(_|\s+)", '', sub).lower()
        rel_n = re.sub(r"(_|\s+)", '', rel).lower()
        obj_n = re.sub(r"(_|\s+)", '', obj).lower()
        return f"{sub_n}{rel_n}{obj_n}"                 # concatenate them to a single string
    
    def _get_subject_object_hallucinations(self, response: dict) -> tuple[float, float]:
        """Calculate subject and object hallucinations metrics, return `SH, OH`. SH/OH check if the 
        tokenized/stemmed subject/object are present in the tokenized/stemmed original sentence or 
        in the tokenized/stemmed ontology concepts."""
        if len(response["triples"]) == 0: return 0, 0

        # append the test sentence with concepts from the ontology
        test_sentence = self.test_sentences[response["id"]]["sent"] + " ".join(self.onto_concepts)  # concat original_sentence|onto_concepts
        stemmed_sentence = "".join([self.ps.stem(word) for word in word_tokenize(test_sentence)])   # stem test_sentence|onto_concepts 
        normalized_stemmed_sentence = re.sub(r"(_|\s+)", '', stemmed_sentence).lower()              # remove white spaces and underscores

        # count the number of subject and object hallucinations
        num_subj_hallucinations, num_obj_hallucinations = 0, 0
        for triple in response["triples"]:
            # clean and normalize subject and object noun phrases the same way as the test sentence
            normalized_stemmed_subject = self._clean_entity_string(triple["sub"])
            normalized_stemmed_object = self._clean_entity_string(triple["obj"])

            # check if the subject/object is found in the stemmed sentence/context text. If not found, mark it as a hallucination
            if normalized_stemmed_sentence.find(normalized_stemmed_subject) == -1: num_subj_hallucinations += 1
            if normalized_stemmed_sentence.find(normalized_stemmed_object) == -1: num_obj_hallucinations += 1

        subj_hallucination = num_subj_hallucinations / len(response["triples"])
        obj_hallucination = num_obj_hallucinations / len(response["triples"])
        return subj_hallucination, obj_hallucination

    def _clean_entity_string(self, entity: str) -> str:
        """Clean subject and object strings of triples by stemming every token from 
        nltk tokenization and removing spaces/underscores and mapping to lowercase"""
        entity = entity.replace("_", " ")                                                    # do not tokenize with underscores !
        stemmed_entity = "".join([self.ps.stem(word) for word in word_tokenize(entity)])     # stem every word for better matches
        normalized_stemmed_entity = re.sub(r"(_|\s+)", '', stemmed_entity).lower()           # remove spaces/underscores and then converting to lower case
        return normalized_stemmed_entity.replace("01januari", "")                            # special handling for string with years to remove January 01

    def _get_ontology_conformance_RH(self, response: dict) -> tuple[float, float]:
        """Compute the OC and RH and relation hallucination metrics, return OC, 1-OC=RH.
        OC is the #of output triples conforming to ontology/#of llm triples, a triple
        conforms to the ontology if relation is one of the canonical relations of the ontology."""
        if len(response["triples"]) == 0: return 1, 0
        # count the number of system triples relations that are in the ontology
        num_rels_conformant = 0
        for triple in response["triples"]:
            clean_rel = "_".join(camelCaseToSpaces(triple["rel"]).split()).lower().strip()
            
            if clean_rel in self.onto_relations: num_rels_conformant += 1
            
        # ontology conformance is the number of system triples relations in the ontology divided by the total number of system triples
        ont_conformance = num_rels_conformant / len(response["triples"])
        # relation hallucination is 1 - ontology conformance
        rel_hallucination = 1 - ont_conformance
        return ont_conformance, rel_hallucination


def generate_global_averages(llm_metrics_folder_path: str):
    avg_files = glob.glob(llm_metrics_folder_path + "/*_avg.jsonl")
    avg_metrics_dbpedia_webnlg = {
        "all": { "n_sentences": 0, "avg_precision": 0, "avg_recall": 0, "avg_f1": 0, "avg_onto_conf": 0, "avg_sub_halluc": 0, "avg_rel_halluc": 0, "avg_obj_halluc": 0}
    }
    
    avg_metrics_wikidata_tekgen = {
        "unseen": { "n_sentences": 0, "avg_precision": 0, "avg_recall": 0, "avg_f1": 0, "avg_onto_conf": 0, "avg_sub_halluc": 0, "avg_rel_halluc": 0, "avg_obj_halluc": 0}, 
        "verified": { "n_sentences": 0, "avg_precision": 0, "avg_recall": 0, "avg_f1": 0, "avg_onto_conf": 0, "avg_sub_halluc": 0, "avg_rel_halluc": 0, "avg_obj_halluc": 0}, 
        "all": { "n_sentences": 0, "avg_precision": 0, "avg_recall": 0, "avg_f1": 0, "avg_onto_conf": 0, "avg_sub_halluc": 0, "avg_rel_halluc": 0, "avg_obj_halluc": 0}
    }

    for avg_file_path in avg_files:
        averages = load_jsonl_as_list(avg_file_path)
        
        for average in averages:
            for typ in average.keys():        
                if typ in ["unseen", "all", "verified"]:
                    for key in average[typ].keys():
                        if "wikidata_tekgen" in avg_file_path:
                            avg_metrics_wikidata_tekgen[typ][key] += average[typ][key] / len(averages)
                        if "dbpedia_webnlg" in avg_file_path and typ == "all":
                            avg_metrics_dbpedia_webnlg[typ][key] += average[typ][key] / len(averages)
    
    with open(llm_metrics_folder_path + "/global_avg.csv", "w") as table_f:
        table_f.write("dataset, subset, P, R, F1, OC, SH, RH, OH\n")
        for typ in avg_metrics_wikidata_tekgen.keys():
            if typ in ["unseen", "all", "verified"]:
                table_f.write(f"wikidata_tekgen, {typ}, {avg_metrics_wikidata_tekgen[typ]['avg_precision']:.2f}, {avg_metrics_wikidata_tekgen[typ]['avg_recall']:.2f}, ")
                table_f.write(f"{avg_metrics_wikidata_tekgen[typ]['avg_f1']:.2f}, {avg_metrics_wikidata_tekgen[typ]['avg_onto_conf']:.2f}, ")
                table_f.write(f"{avg_metrics_wikidata_tekgen[typ]['avg_sub_halluc']:.2f}, {avg_metrics_wikidata_tekgen[typ]['avg_rel_halluc']:.2f}, ")
                table_f.write(f"{avg_metrics_wikidata_tekgen[typ]['avg_obj_halluc']:.2f}\n")
        
        if os.path.exists(llm_metrics_folder_path + "/dbpedia_webnlg_clean_avg.jsonl"):
            for typ in avg_metrics_dbpedia_webnlg.keys():
                if typ in ['unseen', 'all', 'verified']:
                    table_f.write(f"dbpedia_webnlg, {typ}, {avg_metrics_dbpedia_webnlg[typ]['avg_precision']:.2f}, {avg_metrics_dbpedia_webnlg[typ]['avg_recall']:.2f}, ")
                    table_f.write(f"{avg_metrics_dbpedia_webnlg[typ]['avg_f1']:.2f}, {avg_metrics_dbpedia_webnlg[typ]['avg_onto_conf']:.2f}, ")
                    table_f.write(f"{avg_metrics_dbpedia_webnlg[typ]['avg_sub_halluc']:.2f}, {avg_metrics_dbpedia_webnlg[typ]['avg_rel_halluc']:.2f}, ")
                    table_f.write(f"{avg_metrics_dbpedia_webnlg[typ]['avg_obj_halluc']:.2f}\n")


def generate_global_median_quartiles(llm_metrics_folder_path: str):
    mediam_metrics_dbpedia_webnlg = {
        "all": { "precision": [], "recall": [], "f1": [], "onto_conf": [], "sub_halluc": [], "rel_halluc": [], "obj_halluc": []}
    }
    
    mediam_metrics_wikidata_tekgen = {
        "unseen": { "precision": [], "recall": [], "f1": [], "onto_conf": [], "sub_halluc": [], "rel_halluc": [], "obj_halluc": []}, 
        "verified": { "precision": [], "recall": [], "f1": [], "onto_conf": [], "sub_halluc": [], "rel_halluc": [], "obj_halluc": []}, 
        "all": { "precision": [], "recall": [], "f1": [], "onto_conf": [], "sub_halluc": [], "rel_halluc": [], "obj_halluc": []}
    }
    
    for avg_file_path in [llm_metrics_folder_path + "/dbpedia_webnlg_clean_avg.jsonl", llm_metrics_folder_path + "/wikidata_tekgen_avg.jsonl"]:
        if os.path.exists(avg_file_path):
            onto_averages = load_jsonl_as_list(avg_file_path)
            
            for onto_average in onto_averages:
                for variant in onto_average.keys():        
                    if variant in ["unseen", "all", "verified"]:
                        for metric in onto_average[variant].keys():
                            if metric != "n_sentences":
                                if "wikidata_tekgen" in avg_file_path:
                                    mediam_metrics_wikidata_tekgen[variant][metric.replace("avg_", "")].append(onto_average[variant][metric])
                                
                                if "dbpedia_webnlg" in avg_file_path and variant == "all":
                                    mediam_metrics_dbpedia_webnlg[variant][metric.replace("avg_", "")].append(onto_average[variant][metric])

    for variant in mediam_metrics_wikidata_tekgen.keys():
        for metric in mediam_metrics_wikidata_tekgen[variant].keys():
            values = np.array(mediam_metrics_wikidata_tekgen[variant][metric])
            values.sort()
             
            mediam_metrics_wikidata_tekgen[variant][metric] = {
                "median": np.median(values),
                "p-25": np.percentile(values, 25),
                "p-75": np.percentile(values, 75)
            }
    
    if os.path.exists(llm_metrics_folder_path + "/dbpedia_webnlg_clean_avg.jsonl"):
        for variant in mediam_metrics_dbpedia_webnlg.keys():
            for metric in mediam_metrics_dbpedia_webnlg[variant].keys():
                values = np.array(mediam_metrics_dbpedia_webnlg[variant][metric])
                values.sort()
                
                mediam_metrics_dbpedia_webnlg[variant][metric] = {
                    "median": np.median(values),
                    "p-25": np.percentile(values, 25),
                    "p-75": np.percentile(values, 75)
                }

    with open(llm_metrics_folder_path + "/global_median.csv", "w") as table_f:
        for statistic in ["median", "p-25", "p-75"]:
            table_f.write(statistic + "\n")
            table_f.write("dataset, subset, P, R, F1, OC, SH, RH, OH\n")
            for variant in mediam_metrics_wikidata_tekgen.keys():
                if variant in ["unseen", "all", "verified"]:
                    table_f.write(f"wikidata_tekgen, {variant}, {mediam_metrics_wikidata_tekgen[variant]['precision'][statistic]:2f}, {mediam_metrics_wikidata_tekgen[variant]['recall'][statistic]:2f}, ")
                    table_f.write(f"{mediam_metrics_wikidata_tekgen[variant]['f1'][statistic]:2f}, {mediam_metrics_wikidata_tekgen[variant]['onto_conf'][statistic]:2f}, ")
                    table_f.write(f"{mediam_metrics_wikidata_tekgen[variant]['sub_halluc'][statistic]:2f}, {mediam_metrics_wikidata_tekgen[variant]['rel_halluc'][statistic]:2f}, ")
                    table_f.write(f"{mediam_metrics_wikidata_tekgen[variant]['obj_halluc'][statistic]:2f}\n")
            
            if os.path.exists(llm_metrics_folder_path + "/dbpedia_webnlg_clean_avg.jsonl"):
                for variant in mediam_metrics_dbpedia_webnlg.keys():
                    if variant in ['unseen', 'all', 'verified']:
                        table_f.write(f"dbpedia_webnlg, {variant}, {mediam_metrics_dbpedia_webnlg[variant]['precision'][statistic]:2f}, {mediam_metrics_dbpedia_webnlg[variant]['recall'][statistic]:2f}, ")
                        table_f.write(f"{mediam_metrics_dbpedia_webnlg[variant]['f1'][statistic]:2f}, {mediam_metrics_dbpedia_webnlg[variant]['onto_conf'][statistic]:2f}, ")
                        table_f.write(f"{mediam_metrics_dbpedia_webnlg[variant]['sub_halluc'][statistic]:2f}, {mediam_metrics_dbpedia_webnlg[variant]['rel_halluc'][statistic]:2f}, ")
                        table_f.write(f"{mediam_metrics_dbpedia_webnlg[variant]['obj_halluc'][statistic]:2f}\n")

            table_f.write("\n")

def get_csv_avg_per_ontology_dbpedia(llm_response_subfolder: str) -> None:
    """Generate CSV table format from _avg.jsonl files for DBpedia for easy LaTeX or Excel parsing"""
    if os.path.exists(llm_response_subfolder + "/dbpedia_webnlg_clean_avg.jsonl"):    
        with open(llm_response_subfolder + "/dbpedia_webnlg_clean_avg.jsonl") as f:
            data_dbpedia = [json.loads(line) for line in f]
            
            with open(llm_response_subfolder + "/dbpedia_webnlg_clean_avg_per_ontology.csv", "a") as f:
                f.write("onto, P, R, F1, OC, SH, RH, OH\n")
                for ont_result in data_dbpedia:    
                    f.write(f"{ont_result['onto']}, {ont_result['all']['avg_precision']:.2f}, {ont_result['all']['avg_recall']:.2f}, {ont_result['all']['avg_f1']:.2f}, ")
                    f.write(f"{ont_result['all']['avg_onto_conf']:.2f}, {ont_result['all']['avg_sub_halluc']:.2f}, {ont_result['all']['avg_rel_halluc']:.2f}, {ont_result['all']['avg_obj_halluc']:.2f}\n")
    else:
        print("No DBpedia_WebNLG averages at", llm_response_subfolder + "/dbpedia_webnlg_clean_avg.jsonl")


def get_csv_avg_per_ontology_tekgen(llm_response_subfolder: str) -> None:
    """Generate CSV table format from _avg.jsonl files for TekGen for easy LaTeX or Excel parsing"""
    if os.path.exists(llm_response_subfolder + "/wikidata_tekgen_avg.jsonl"):
        with open(llm_response_subfolder + "/wikidata_tekgen_avg.jsonl") as f:
            data_wikidata = [json.loads(line) for line in f]
            for variant in ["all", "unseen", "verified"]:
                with open(llm_response_subfolder + f"/wikidata_tekgen_avg_per_ontology_{variant}.csv", "a") as f:
                    f.write("onto, P, R, F1, OC, SH, RH, OH\n")
                    for ont_result in data_wikidata:
                        f.write(f"{ont_result['onto']}, {ont_result[variant]['avg_precision']:.2f}, {ont_result[variant]['avg_recall']:.2f}, {ont_result[variant]['avg_f1']:.2f}, ")
                        f.write(f"{ont_result[variant]['avg_onto_conf']:.2f}, {ont_result[variant]['avg_sub_halluc']:.2f}, {ont_result[variant]['avg_rel_halluc']:.2f}, {ont_result[variant]['avg_obj_halluc']:.2f}\n")
    else:
        print("No Wikidata-TekGen averages at", llm_response_subfolder + "/wikidata_tekgen_avg.jsonl")

if __name__ ==  "__main__":
    
    llm_response_folders = glob.glob("../results/llm_responses/*")
    print(llm_response_folders)
    for llm_response_folder_path in llm_response_folders: 
        #llm_response_folder_path = "rebel-fine-tuned-per-ontology-01-january-checkpoints"
        #if "rel-in-ontology" not in llm_response_folder_path: continue

        print("Compute metrics for", llm_response_folder_path)
        folder_name = llm_response_folder_path.split("/")[-1]
        
        if os.path.exists("../results/metrics/" + folder_name): continue

        for ontology_name in WIKIDATA_TEKGEN_ONT_NAMES:
            print("Wikidata-TekGen", ontology_name)
            l = LLMMetrics(folder_name, ontology_name, "wikidata_tekgen")
            l.generate()
        
        # if one dbpedia file exists, assume they all do
        if os.path.exists("../results/llm_responses/" + folder_name + "/ont_1_university-dbpedia_webnlg_clean.jsonl"):
            for ontology_name in DBPEDIA_WEBNLG_ONT_NAMES:
                print("DBpedia-WebNLG", ontology_name)
                l = LLMMetrics(folder_name, ontology_name, "dbpedia_webnlg_clean")
                l.generate()     
    
        llm_response_metrics_folder_path = "../results/metrics/" + folder_name
        generate_global_averages(llm_response_metrics_folder_path)
        get_csv_avg_per_ontology_dbpedia(llm_response_metrics_folder_path)
        get_csv_avg_per_ontology_tekgen(llm_response_metrics_folder_path)  
        generate_global_median_quartiles(llm_response_metrics_folder_path)
