from prompter import Prompter
from adapter import LLMAdapter
from typing import List
import json

class ExperimentConfig:
    """
    Simple structure to contain experiment configuration, parameters are, 
    - `train_file_path` path towards the train sentences `jsonl` file
    - `test_file_path` path towards the test sentences `jsonl` file
    - `ontology_file_path` path towards the ontology definition of the train/test files
    - `n_train_examples` number of examples to provide in the prompt for in context learning
    - `experiment_name` for example, wikidata_tekgen_5_train_examples_vicuna13B 
    """
    def __init__(self, train_file_path: str, test_file_path: str, ontology_file_path: str, n_train_examples: str, experiment_name: str):
        self.train_file_path = train_file_path
        self.test_file_path = test_file_path
        self.ontology_file_path = ontology_file_path
        self.experiment_name = experiment_name
        self.n_train_examples = n_train_examples

class Experiment:
    def __init__(self, adapter: LLMAdapter, config: ExperimentConfig):
        self.llm_responses = []
        self.adapter = adapter
        self.config = config
        self.prompter = Prompter(self.config.ontology_file_path, self.config.train_file_path, self.config.test_file_path)
        
        with open(config.test_file_path, "r") as test_sent_f:
            self.test_sentence_ids: List[str] = [json.loads(line)['id'] for line in test_sent_f]
    
    def run(self):
        for test_id in self.test_sentence_ids:
            print("Querying " + test_id)
            self.llm_responses.append(
                self.adapter.queryLLM(
                    test_id, 
                    self.prompter.getPromptOf(test_id, self.config.n_train_examples)
                )
            )
        print("Done.")
        