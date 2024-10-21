from openai import OpenAI
from typing import override


class LLMResponse:
    def __init__(self, sent_id: str, response: str, triples: list):
        self.sent_id = sent_id
        self.response = response
        self.triples = triples
            

class LLMAdapter:
    """
    This interface allows prompting an extracting responses from a model.
    """
    def __init__(self):
        """
        Configure the LLM, for Vicuna and Alpaca it'll check if 
        "../models/model_name" exists and wether the model files 
        are there, if they're not they'll be downloaded from Hugging Face.
        
        For our own REBEL variant the same principle applies, though
        not sure if we'll upload it to Hugging Face once we have it. 
        
        For OpenAI models, only an API-key and model names are required. 
        """
        raise NotImplementedError()
    
    def queryLLM(self, sent_id: str, prompt: str) -> LLMResponse:
        """
        For our own custom models, the method `queryLLM` allows us to 
        implement our own logic as to how to treat the prompt and feed
        it to the model / map it's response back to an `LLMResponse` object.    
        """
        raise NotImplementedError()
    
    
    
class OpenAIAdapter(LLMAdapter):
    @override
    def __init__(self, openai_key: str, model_name: str):
        pass
    
    def queryLLM(self, sent_id: str, prompt: str) -> LLMResponse:
        pass