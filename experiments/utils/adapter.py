from openai import OpenAI
from typing import override
import re

class LLMResponse(object):
    def __init__(self, sent_id: str, response: str, triples: list):
        self.sent_id = sent_id
        self.response = response
        self.triples = triples
            

class LLMAdapter:
    """
    This interface allows prompting an extracting responses from a model.
    """
    def __init__(self, model_name: str):
        """
        Configure the LLM, for Vicuna and Alpaca it'll check if 
        "../models/model_name" exists and wether the model files 
        are there, if they're not they'll be downloaded from Hugging Face.
        
        For our own REBEL variant the same principle applies, though
        not sure if we'll upload it to Hugging Face once we have it. 
        
        For OpenAI models, only an API-key and model names are required. 
        """
        self.model_name = model_name

    def queryLLM(self, sent_id: str, prompt: str) -> LLMResponse:
        """
        For our own custom models, the method `queryLLM` allows us to 
        implement our own logic as to how to treat the prompt and feed
        it to the model / map it's response back to an `LLMResponse` object.    
        """
        raise NotImplementedError()
    
    def getTriplesOf(self, response: str) -> list[dict[str, str]]: 
        triples_raw_strings = response.split("\n")
        triples: list[dict[str, str]] = []
        
        for triple_str in triples_raw_strings:
            # Apply regex
            match = re.match(r"(.+)\((.+),(.+)\)", triple_str)
            if match:
                relation, subject, object_ = match.groups()
                triples.append({"sub": subject.strip(), "rel": relation.strip(), "obj": object_.strip()})

        return triples


class OpenAIAdapter(LLMAdapter):
    def __init__(self, openai_key: str, model_name: str):
        super().__init__(model_name)
        self.client = OpenAI(api_key=openai_key)
        self.model_name = model_name
    
    @override
    def queryLLM(self, sent_id: str, prompt: str) -> LLMResponse:
        chat_completion = self.client.chat.completions.create(
            model=self.model_name,
            messages=[{"role": "user", "content": prompt}],
            timeout=5
        )
        return LLMResponse(
            sent_id, 
            chat_completion.choices[0].message.content.strip(),
            self.getTriplesOf(chat_completion.choices[0].message.content.strip()) 
        )
