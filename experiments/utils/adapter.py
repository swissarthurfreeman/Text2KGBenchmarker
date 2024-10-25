import time

t = time.localtime()
print("Import transformers...", t)
from transformers import pipeline   # this takes ~2:30 on yggdrasil, why ?
print("import openai...", time.localtime())
from openai import OpenAI
print("Time is now :", time.localtime())
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
    
    def queryLLM(self, sent_id: str, prompt: str) -> LLMResponse:
        chat_completion = self.client.chat.completions.create(
            model=self.model_name,
            messages=[{"role": "user", "content": prompt}],
            timeout=10
        )
        return LLMResponse(
            sent_id, 
            chat_completion.choices[0].message.content.strip(),
            self.getTriplesOf(chat_completion.choices[0].message.content.strip()) 
        )


class RebelAdapter(LLMAdapter):
    def __init__(self, model_name: str):
        self.model_name = model_name.replace("/", ".")
        print("Loading Rebel Pipeline...", time.localtime())
        self.rebel_pipeline = pipeline('text2text-generation', model=model_name, tokenizer=model_name, device="cuda")
        print("Done, device is ", self.rebel_pipeline.device, time.localtime())
    
    def queryLLM(self, sent_id: str, prompt: str) -> LLMResponse:
        test_sentence = prompt[prompt.find("Test Sentence:")+len("Test Sentence:"):]
        
        print(test_sentence)
        # returns a list [{"generated_token_ids": tensor([0, 5205, ...])}, ...] for every element of list of inputs passed to pipeline, in this case just one.
        raw_token_idx_tensors = self.rebel_pipeline(test_sentence, return_tensors=True, return_text=False)[0]['generated_token_ids']
        print(raw_token_idx_tensors.device)
        extracted_text: list[str] = self.rebel_pipeline.tokenizer.batch_decode([raw_token_idx_tensors])
        extracted_triples = self.getTriplesOf(extracted_text[0])
        
        return LLMResponse(
            sent_id,
            extracted_text[0],
            extracted_triples
        )
    
    def getTriplesOf(self, response: str) -> list[dict[str, str]]:
        triplets: list[dict[str, str]] = []
        relation, subject, relation, object_ = '', '', '', ''
        text = response.strip()
        current = 'x'
        for token in text.replace("<s>", "").replace("<pad>", "").replace("</s>", "").split():
            if token == "<triplet>":
                current = 't'
                if relation != '':
                    triplets.append({'sub': subject.strip(), 'rel': relation.strip(),'obj': object_.strip()})
                    relation = ''
                subject = ''
            elif token == "<subj>":
                current = 's'
                if relation != '':
                    triplets.append({'sub': subject.strip(), 'rel': relation.strip(),'obj': object_.strip()})
                object_ = ''
            elif token == "<obj>":
                current = 'o'
                relation = ''
            else:
                if current == 't':
                    subject += ' ' + token
                elif current == 's':
                    object_ += ' ' + token
                elif current == 'o':
                    relation += ' ' + token
        if subject != '' and relation != '' and object_ != '':
            triplets.append({'sub': subject.strip(), 'rel': relation.strip(),'obj': object_.strip()})
        return triplets


if __name__ == '__main__':
    rebel = RebelAdapter("Babelscape/rebel-large")
    response = rebel.queryLLM("bogus_id", "Test Sentence: Carouge is a municipality in Geneva, Switzerland.")
    print(response.triples, time.localtime())
    
    response = rebel.queryLLM("bogus_id", "Test Sentence: Pully is a municipality in the canton of Vaud, Switzerland.")
    print(response.triples, time.localtime())