# Utility script to generate sentences from triples by building
# a prompt for an OpenAI GPT. A random, connex subset of the 
# available triples is chosen to provide in the prompt for the model. 
import os
import sys
import json
import random
from openai import OpenAI
from multiprocessing.pool import ThreadPool


class SentenceGenerator:
    """
    >>> samples = [
            {"triples": [
                {"sub": "Violent Silences", "sqid": "Q7933178", "rel": "publication date", "rpid": "P577", ..., "oqid": "2004-01-01T00:00:00Z"}, 
                {"sub": "Violent Silences", "sqid": "Q7933178", "rel": "performer", "rpid": "P175", "obj": "Rico", "oqid": "Q7332245"}
            ]},
            {"triples": [
                {"sub": "Get Some Sleep", "sqid": "Q5554174", "rel": "part of", "rpid": "P361", "obj": "Beautiful Collision", "oqid": "Q4877686"}, 
                {"sub": "Beautiful Collision", "sqid": "Q4877686", "rel": "publication date", "rpid": "P577", ..., "oqid": "2002-01-01T00:00:00Z"}, 
                {"sub": "Beautiful Collision", "sqid": "Q4877686", "rel": "genre", "rpid": "P136", "obj": "pop music", "oqid": "Q37073"}, 
                {"sub": "Beautiful Collision", "sqid": "Q4877686", "rel": "performer", "rpid": "P175", "obj": "Bic Runga", "oqid": "Q467035"}
            ]}
        ]
    >>> gen = SentenceGenerator(samples, '', 'musical works', 'ont_2_music')
    >>> gen.generate(stdout=True)
    T0 Getting prompt and triples for ont_2_music_train_Q7933178
    {
        "id": "ont_2_music_train_Q7933178", 
        "sent": '''Released on January 1, 2004, Violent Silences is a notable musical work performed by Rico. This project showcases Rico\u2019s unique style and 
        musical vision, contributing to the broader landscape of music from the early 2000s.''', 
        "triples": [
            {"sub": "Violent Silences", "sqid": "Q7933178", "rel": "publication date", "rpid": "P577", ..., "oqid": "2004-01-01T00:00:00Z"}, 
            {"sub": "Violent Silences", "sqid": "Q7933178", "rel": "performer", "rpid": "P175", "obj": "Rico", "oqid": "Q7332245"}
        ]
    }
    T0 Getting prompt and triples for ont_2_music_train_Q5554174
    {
        "id": "ont_2_music_train_Q5554174", 
        "sent": '''Get Some Sleep is a standout track from Bic Runga's album Beautiful Collision, which was released on January 1, 2002. This work is primarily categorized 
        within the pop music genre, showcasing Runga's distinctive style and lyrical depth. The album has continued to resonate with fans, maintaining its relevance in the 
        pop music scene years after its debut.''', 
        "triples": [
            {"sub": "Get Some Sleep", "sqid": "Q5554174", "rel": "part of", "rpid": "P361", "obj": "Beautiful Collision", "oqid": "Q4877686"}, 
            {"sub": "Beautiful Collision", "sqid": "Q4877686", "rel": "publication date", "rpid": "P577", ..., "oqid": "2002-01-01T00:00:00Z"}, 
            {"sub": "Beautiful Collision", "sqid": "Q4877686", "rel": "genre", "rpid": "P136", "obj": "pop music", "oqid": "Q37073"}, 
            {"sub": "Beautiful Collision", "sqid": "Q4877686", "rel": "performer", "rpid": "P175", "obj": "Bic Runga", "oqid": "Q467035"}
        ]
    }
    """
    def __init__(self, samples: list[dict[str, str|list[dict]]], output_file_path: str, prompt_subject: str, ont_name: str):
        self.prompt_subject =  prompt_subject
        self.output_file_path = output_file_path
        self.samples = samples
        self.ont_name = ont_name
    
    def generate(self, i: int = 0, stdout: bool = False):
        client = OpenAI(api_key="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")

        for sample in self.samples:
            sample['id'] = f"{self.ont_name}_train_{sample['triples'][0]['sqid']}"
            
            print(f"T{i} Getting prompt and triples for", sample['id'])
            prompt, chosen_triples = self.getPrompt(sample['triples'])
            
            chat_completion = client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[{"role": "user", "content": prompt}],
                timeout=10
            )
            
            res = {'id': sample['id'], 'sent': chat_completion.choices[0].message.content.strip(), 'triples': chosen_triples}

            if stdout: 
                print(json.dumps(res))
                continue
            
            with open(output_file_path, "a", encoding='utf-8') as f:
                f.write(json.dumps(res) + "\n")

    def getPrompt(self, triples: list[dict]) -> tuple[str, list[dict]]:
        prompt = f"""Your task is to generate varied, accurate, sentences about {self.prompt_subject} from a list of triples following these precise instructions :
        1) Make the sentences natural, as they would appear in text or conversation, wikipedia, news or the internet.
        2) The triples can be explicit in the sentence or implied, include all the triples provided, do not invent additional facts. 
        3) If no currency or timezone are specified, do not include them.
        4) Do not put names or titles between quotation marks like " or ' unless absolutely necessary.
        5) Use varied turn of phrases, don't always start the sentence with the name of the main subject, vary using coreference instead of repeating the title. 
        6) Vary the output size, sometimes a single sentence, sometimes a small paragraph, not more than a couple of sentences.
        Triples:\n
        """
        triples = self.getRandomNTriplesFrom(triples)
        prompt += self.getTextTriplesList(triples)
        return prompt, triples

    def getTextTriplesList(self, triples: list[dict]) -> str:
        res = ""
        for triple in triples:
            res += triple['rel'] + "(" + triple['sub'] + " <SEP> " + triple['obj'] + ")\n"
        return res

    def getRandomNTriplesFrom(self, triples: list[dict], max_n: int = 9) -> list[dict]:
        """Here we do custom random BFS like graph exploration from the first triple to generate a 
        list of n connected triples. No need for node labeling as graph has no cycles."""
        res: list[dict] = []
        queue: list[dict] = [triples[0]]
        
        curr = queue.pop(0)
        res.append(curr)
        
        n_triples = random.randint(3, max_n)
        if len(triples) < n_triples: return triples     # more triples requested than available, return all triples
        
        while len(res) != n_triples and len(queue) > 0:
            queue += [triple for triple in triples if ( triple['sqid'] == curr['sqid'] or curr['oqid'] == triple['sqid']) and triple not in res]
            random.shuffle(queue)
            curr = queue.pop(0)
            if curr not in res: res.append(curr)
        
        return res


def getSamplesWithoutSentencesInOutputFile(raw_triples_path: str, output_file_path: str) -> list[dict]:
    qids_in_output_file = []

    if os.path.exists(output_file_path):
        with open(output_file_path) as f:   # get all root qids which have been verbalized
            qids_in_output_file = [json.loads(line)['id'].split("_")[-1] for line in f]
            
    raw_triples = []
    with open(raw_triples_path) as f:   # get all triples in raw_triples
        raw_triples = [json.loads(line) for line in f]
    
    # only keep those that haven't been verbalized yet
    return [sample for sample in raw_triples if sample['triples'][0]['sqid'] not in qids_in_output_file]


def worker(i, samples, output_file_path, prompt_subject, ont_name):
    print(f"T{i} deals with", len(samples), "samples, 1st sample triple sqid", samples[0]['triples'][0]['sqid'])
    
    print()
    gen = SentenceGenerator(samples, output_file_path, prompt_subject, ont_name)
    gen.generate(stdout=False)

if __name__ == "__main__":
    ontology_name    = sys.argv[1]  # 'ont_2_music'
    prompt_subject   = sys.argv[2]  # 'musical works'
    raw_triples_path = sys.argv[3]  # './raw_triples/ont_2_music_triples.jsonl 
    output_file_path = sys.argv[4]  # './some_path/ont_2_music_train.jsonl'
    
    samples = getSamplesWithoutSentencesInOutputFile(raw_triples_path, output_file_path)

    print("Found", len(samples), "that haven't been verbalized...")

    n_threads = 4
    pool = ThreadPool(n_threads)

    for i in range(n_threads):
        pool.apply_async(worker, (i, samples[i*(len(samples)//n_threads):(i+1)*(len(samples)//n_threads)], output_file_path, prompt_subject, ontology_name))
    
    pool.close()
    pool.join()