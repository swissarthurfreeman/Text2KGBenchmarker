# Utility script to generate sentences from triples by building
# a prompt for an OpenAI GPT. A random, connex subset of the 
# available triples is chosen to provide in the prompt for the model. 

from multiprocessing.pool import ThreadPool
from dotenv import load_dotenv
from openai import OpenAI
load_dotenv()
import random
import json
import sys
import os


def getTextTriplesList(triples: list[dict]) -> str:
    res = ""
    for triple in triples:
        res += triple['rel'] + "(" + triple['sub'] + " <SEP> " + triple['obj'] + ")\n"
    return res

def toSet(triples: list[dict]) -> set[str]:
    res = set()
    for triple in triples:
        res.add(triple['rel'] + "(" + triple['sub'] + "<SEP>" + triple['obj'] + ")")
    return res

def getRandomNTriplesFrom(triples: list[dict], max_n: int) -> list[dict]:
    """Here we do custom random BFS like graph exploration from the first triple to generate a 
    list of n connected triples. No need for node labeling as graph has no cycles."""
    res: list[dict] = []
    queue: list[dict] = [triples[0]]
    
    curr = queue.pop(0)
    res.append(curr)
    
    if len(triples) < max_n:
        return triples
    
    while len(res) != max_n and len(queue) > 0:
        queue += [triple for triple in triples if ( triple['sqid'] == curr['sqid'] or triple['sqid'] == curr['oqid']) and triple not in res]
        random.shuffle(queue)
        
        curr = queue.pop(0)
        if curr not in res:
            res.append(curr)
    return res

def getPrompt(triples: list[dict], subject: str) -> tuple[str, list[dict]]:
    prompt = f"""Your task is to generate varied, accurate, sentences about {subject} from a list of triples following these precise instructions :
1) Make the sentences natural, as they would appear in text or conversation, wikipedia, news or the internet.
2) The triples can be explicit in the sentence or implied, include all the triples provided, do not invent additional facts. 
3) If no currency or timezone are specified, do not include them.
4) Do not put names or titles between quotation marks like " or ' unless absolutely necessary.
5) Use varied turn of phrases, don't always start the sentence with the name of the main subject, vary using coreference instead of repeating the title. 
6) Vary the output size, sometimes a single sentence, sometimes a small paragraph, not more than a couple of sentences.
Triples:\n
"""
    n_triples = random.randint(3, 9)
    triples = getRandomNTriplesFrom(triples, n_triples)
    prompt += getTextTriplesList(triples)
    return prompt, triples


def worker(i, raw_triples, output_file_path, subject):
    print(f"T{i} deals with", len(raw_triples), "triples, first triple sqid", raw_triples[0]['triples'][0]['sqid'])
    client = OpenAI(api_key="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")
    
    for line in raw_triples:
        line['id'] = f"ont_2_music_train_{line['triples'][0]['sqid']}"
        
        if line['id'] in ids_already_in_output_file:
            continue
        
        print(f"T{i} Getting prompt and triples for", line['id'])
        prompt, chosen_triples = getPrompt(line['triples'], subject)
        
        chat_completion = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[{"role": "user", "content": prompt}],
            timeout=10
        )
        
        res = {'id': line['id'], 'sent': chat_completion.choices[0].message.content.strip(), 'triples': chosen_triples}
        with open(output_file_path, "a", encoding='utf-8') as f:
            f.write(json.dumps(res) + "\n")
      

if __name__ == "__main__":
    
    triples_file_path = sys.argv[-3] # "ont_2_music.jsonl" 
    output_file_path = sys.argv[-2]  # "ont_2_music_train.jsonl"
    subject = sys.argv[-1]
    
    ids_already_in_output_file = []
    with open(output_file_path) as f:
        ids_already_in_output_file = [json.loads(line)['id'].split("_")[-1] for line in f]
        
    
    raw_triples = []
    with open(triples_file_path) as f:
        raw_triples = [json.loads(line) for line in f]
    
    raw_triples = [line for line in raw_triples if line['triples'][0]['sqid'] not in ids_already_in_output_file]
    
    n_threads = 4
    pool = ThreadPool(n_threads)
    
    for i in range(n_threads):
        pool.apply_async(worker, (i, raw_triples[i*(len(raw_triples)//n_threads):(i+1)*(len(raw_triples)//n_threads)], output_file_path, subject))
    
    pool.close()
    pool.join()