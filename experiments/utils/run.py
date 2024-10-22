from prompter import Prompter
from adapter import LLMAdapter, LLMResponse
import os
import json

class LLMRunConfig:
    """
    Simple structure to contain run configuration, parameters are, 
    - `train_file_path` path towards the train sentences `jsonl` file
    - `test_file_path` path towards the test sentences `jsonl` file
    - `ontology_file_path` path towards the ontology definition of the train/test files
    - `adapter` LLMAdapter instance, with loaded LLM configuration
    """
    def __init__(self, train_file_path: str, test_file_path: str, ontology_file_path: str, n_train_examples: int, adapter: LLMAdapter):
        self.train_file_path = train_file_path
        self.test_file_path = test_file_path
        self.ontology_file_path = ontology_file_path
        self.n_train_examples = n_train_examples
        self.adapter = adapter

class LLMRun:
    """
    Given a LLMRunConfig, run inference for all test sentences according
    to the configuration object. Save llm responses to results folder. 
    """
    def __init__(self, config: LLMRunConfig):
        self.llm_responses: list[LLMResponse] = []
        self.config = config
        self.prompter = Prompter(self.config.ontology_file_path, self.config.train_file_path, self.config.test_file_path)
        
        with open(config.test_file_path, "r") as test_sent_f:
            self.test_sentence_ids: list[str] = [json.loads(line)['id'] for line in test_sent_f]
    
    def run(self):
        print("Starting LLM run for ontology at: " + self.config.ontology_file_path + "\nwith model: " + self.config.adapter.model_name + "\nn_train_examples: " + str(self.config.n_train_examples))
        res_file_path = self.resolveResultsFilePath()
        print("Saving responses to " + res_file_path)
        
        # if file already exist, picup from where we left off
        start_idx = 0
        if os.path.exists(res_file_path):
            with open(res_file_path, "r") as f:    
                start_idx = len(f.readlines())
                if start_idx > 0: print("Picking up at line n°" + str(start_idx+1))
        
        for test_id in self.test_sentence_ids[start_idx:]:
            print("Querying " + test_id)
            response = self.config.adapter.queryLLM(
                    test_id, 
                    self.prompter.getPromptOf(test_id, self.config.n_train_examples)
                )
            
            with open(res_file_path, "a") as f:
                f.write(json.dumps(vars(response)) + "\n")
        
        print("Done")
        
    def resolveResultsFilePath(self) -> str:
        """For example, ../results/llm_responses/gpt-4o/1_movie_ontology.test_wikidata_tekgen.train_wikidata_tekgen.n_examples_4"""
        res_file_path = ["../results/llm_responses/" + self.config.adapter.model_name + "/"]
        
        if not os.path.exists("../results/llm_responses/" + self.config.adapter.model_name):
            os.makedirs("../results/llm_responses/" + self.config.adapter.model_name)
        
        # append ontology name
        res_file_path.append(self.config.ontology_file_path.split("/")[-1][:-5])
        
        res_file_path = ["".join(res_file_path)]
        
        if "wikidata_tekgen" in self.config.test_file_path: res_file_path.append("test_wikidata_tekgen")
        if "wikidata_tekgen" in self.config.train_file_path: res_file_path.append("train_wikidata_tekgen")
        
        if "dpedia_webnlg" in self.config.test_file_path: res_file_path.append("test_dpedia_webnlg")
        if "dpedia_webnlg" in self.config.train_file_path: res_file_path.append("train_dpedia_webnlg")
        
        res_file_path.append("n_examples_" + str(self.config.n_train_examples))
        return "-".join(res_file_path) + ".jsonl"
        
        