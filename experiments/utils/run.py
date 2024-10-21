from prompter import Prompter
from adapter import LLMAdapter, LLMResponse
import json

class LLMRunConfig:
    """
    Simple structure to contain run configuration, parameters are, 
    - `train_file_path` path towards the train sentences `jsonl` file
    - `test_file_path` path towards the test sentences `jsonl` file
    - `ontology_file_path` path towards the ontology definition of the train/test files
    - `experiment_name` for example, wikidata_tekgen_5_train_examples_vicuna13B 
    - `adapter` LLMAdapter instance, with loaded LLM configuration
    """
    def __init__(self, train_file_path: str, test_file_path: str, ontology_file_path: str, n_train_examples: str, adapter: LLMAdapter):
        self.train_file_path = train_file_path
        self.test_file_path = test_file_path
        self.ontology_file_path = ontology_file_path
        self.n_train_examples = n_train_examples
        self.adapter = adapter

class LLMRun:
    def __init__(self, config: LLMRunConfig):
        self.llm_responses: list[LLMResponse] = []
        self.config = config
        self.prompter = Prompter(self.config.ontology_file_path, self.config.train_file_path, self.config.test_file_path)
        
        with open(config.test_file_path, "r") as test_sent_f:
            self.test_sentence_ids: list[str] = [json.loads(line)['id'] for line in test_sent_f]
    
    def run(self):
        for test_id in self.test_sentence_ids:
            print("Querying " + test_id)
            self.llm_responses.append(
                self.adapter.queryLLM(
                    test_id, 
                    self.prompter.getPromptOf(test_id, self.config.n_train_examples)
                )
            )
        self.save_responses()
        print("Done.")
        
    def save_responses(self) -> None:
        with open("../results/llm_responses/" + self.config.adapter):
            for response in self.llm_responses:
                pass
            
        