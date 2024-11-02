from transformers import AutoConfig, AutoModelForSeq2SeqLM, AutoTokenizer, pipeline
from pprint import pprint
from openai import OpenAI
from time import time
import re

class LLMResponse(object):
    def __init__(self, sent_id: str, response: str, triples: list, time: float = None):
        self.id = sent_id
        self.response = response
        self.triples = triples
        if time != None:
            self.time = time
            """seconds it took to run the inference"""
            

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
            match = re.match(r"([^()]+)\((.+)\|(.+)\)", triple_str)
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
    def __init__(self, model_name: str, device: str = "cuda", n_beams: int = 1, n_return_sequences: int = 1):
        self.model_name = model_name.replace("/", ".")
        
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForSeq2SeqLM.from_pretrained(model_name).to(device)
        
        self.gen_kwargs = {
            "max_length": 256,
            "length_penalty": 0,
            "num_beams": n_beams,
            "num_return_sequences": n_return_sequences,
        }
        
    def queryLLM(self, sent_id: str, prompt: str) -> LLMResponse:
        start = time()
        test_sentence = prompt[prompt.find("Test Sentence:")+len("Test Sentence:"):]
        model_inputs = self.tokenizer(test_sentence, padding=True, truncation=True, return_tensors='pt')
        generated_tokens = self.model.generate(
            model_inputs["input_ids"].to(self.model.device),
            attention_mask=model_inputs["attention_mask"].to(self.model.device),
            **self.gen_kwargs
        )

        decoded_preds = self.tokenizer.batch_decode(generated_tokens, skip_special_tokens=False)
        triples = [] 
        for beam in decoded_preds:
            triples += self.getTriplesOf(beam)
        
        return LLMResponse(
            sent_id,
            "".join(decoded_preds),
            triples,
            time() - start 
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
    rebel = RebelAdapter("Babelscape/rebel-large", "cpu", 4, 2)
    response: LLMResponse = rebel.queryLLM("bogus_id", "Test Sentence: Carouge is a municipality in Geneva, Switzerland.")
    print(response.triples)
    
    response: LLMResponse = rebel.queryLLM("bogus_id", "Test Sentence: Pully is a municipality in the canton of Vaud, Switzerland.")
    print(response.triples)