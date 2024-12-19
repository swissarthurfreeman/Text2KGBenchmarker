import os
import json
from prompter import Prompter
from adapter import LLMAdapter, LLMResponse, OpenAIAdapter, RebelAdapter
from utils import WIKIDATA_TEKGEN_ONT_NAMES, DPEDIA_WEBNLG_ONT_NAMES, load_jsonl_as_list


class LLMRunConfig:
    """
    Simple structure to contain run configuration, parameters are, 
    - `train_file_path` path towards the train sentences `jsonl` file
    - `test_file_path` path towards the test sentences `jsonl` file
    - `ontology_file_path` path towards the ontology definition of the train/test files
    - `adapter` LLMAdapter instance, with loaded LLM configuration
    """
    def __init__(self, dataset_name: str, ontology_name: str, adapter: LLMAdapter, n_train_examples: int | None = None, n_beams: int | None = None):
        self.dataset_name = dataset_name
        self.ontology_name = ontology_name
        self.adapter = adapter
        self.n_train_examples = n_train_examples
        self.n_beams = n_beams
        

class LLMRun:
    """
    Given a LLMRunConfig, run inference for all test sentences according
    to the configuration object. Save llm responses to results folder. 
    """
    def __init__(self, config: LLMRunConfig):
        self.llm_responses: list[LLMResponse] = []
        self.config = config
        self.prompter = Prompter(self.config.dataset_name, self.config.ontology_name)
        self.test_sentence_ids = list(self.prompter.test_sentences.keys())
        
        self.llm_response_dir = "../results/llm_responses/" + self.config.adapter.model_name
        if self.config.n_train_examples != None:
            self.llm_response_dir += "-" + str(self.config.n_train_examples) + "-shot"
        
        if self.config.n_beams != None:
            self.llm_response_dir += "-" + str(self.config.n_beams) + "-beams"
        
        if not os.path.exists(self.llm_response_dir): 
            os.makedirs(self.llm_response_dir)
    
    def run(self):
        """Run inference for all test ids of configuration, pick-up at last inference if interrupted."""
        print("Starting LLM run for ontology : " + self.config.ontology_name + "\ndataset : " + self.config.dataset_name + "\nwith model: " + self.config.adapter.model_name + "\nn_train_examples: " + str(self.config.n_train_examples))
        
        res_file_path = self.llm_response_dir + "/" + self.config.ontology_name + "-" + self.config.dataset_name + ".jsonl"
         
        print("Saving responses to " + res_file_path)
        
        # if file already exist, pick-up from where we left off
        start_idx = 0
        if os.path.exists(res_file_path):
            with open(res_file_path, "r") as f:    
                start_idx = len(f.readlines())
                if start_idx > 0: print("Picking up at line n°" + str(start_idx+1))
        
        for test_id in self.test_sentence_ids[start_idx:]:
            print("Querying " + test_id)
            prompt = self.prompter.getPromptOf(test_id, self.config.n_train_examples)
            
            k = int( test_id.split("_")[-1] )
            if k % 50 == 0:
                print(prompt)
            
            response: LLMResponse = self.config.adapter.queryLLM(
                    test_id, 
                    prompt
                )
            
            with open(res_file_path, "a") as f:
                f.write(json.dumps(vars(response)) + "\n")
        
        print("done")


if __name__ == "__main__":
    def run_inference_on(dataset_name: str, ontology_name: str, adapter: LLMAdapter, i: int):
            conf = LLMRunConfig(
                dataset_name=dataset_name,
                ontology_name=ontology_name,
                adapter=adapter,
                n_train_examples=i
            )
            
            runner = LLMRun(conf)
            runner.run()
    
    for i in [1, 2, 3, 4, 5, 6]:
        model_adapter = OpenAIAdapter("", "gpt-4o")
        
        for ontology_name in DPEDIA_WEBNLG_ONT_NAMES:
            run_inference_on("dpedia_webnlg_clean", ontology_name, model_adapter, i)
        
        for ontology_name in WIKIDATA_TEKGEN_ONT_NAMES:
            run_inference_on("wikidata_tekgen", ontology_name, model_adapter, i)
        
    
        
