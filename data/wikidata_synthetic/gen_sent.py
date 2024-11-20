# Utility script to generate sentences from triples by building
# a prompt for an OpenAI GPT. A random, connex subset of the 
# available triples is chosen to provide in the prompt for the model. 
from openai import OpenAI
import random
import json
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
    
    while len(res) != max_n:
        queue += [triple for triple in triples if ( triple['sqid'] == curr['sqid'] or triple['sqid'] == curr['oqid']) and triple not in res]
        random.shuffle(queue)
        
        curr = queue.pop(0)
        if curr not in res:
            res.append(curr)
    return res

def getPrompt(triples: list[dict]) -> tuple[str, list[dict]]:
    prompt = f"""Please generate sentences on the subject of music from the triples below following these precise instructions :
1) Make the sentences natural, as they would appear in text or conversation.
2) The triples can be explicit in the sentence or implied, include all the triples.
3) If no currency or timezone are specified, do not include them.
4) Do not put the musical work title between quotation marks like " or ' unless absolutely necessary.
5) Use varied turn of phrases, for example, don't always start the sentence with the name of musical work, and use coreference instead of repeating the title. 
6) Make the output no longer than a small paragraph.

Triples:\n
"""
    n_triples = random.randint(3, 12)
    triples = getRandomNTriplesFrom(triples, n_triples)
    prompt += getTextTriplesList(triples)
    return prompt, triples

def getIdx():
    """TODO : clean this up, not needed anymore."""
    with open("./ont_2_music_train.jsonl") as f:
        return len(f.readlines())

if __name__ == "__main__":
    
    data = []
    with open("./ont_2_music.jsonl") as f:
        data = [json.loads(line) for line in f]

    print("Querying GPT...")
    client = OpenAI(api_key=os.environ['OPEN_API_KEY'])
    
    idx = getIdx()
    
    for line in data[idx:]:
        print("Getting prompt and triples for", line['id'])
        prompt, chosen_triples = getPrompt(line['triples'])
        print("Prompting...")
        chat_completion = client.chat.completions.create(
            model="gpt-4o",
            messages=[{"role": "user", "content": prompt}],
            timeout=10
        )
        
        res = {'id': line['id'], 'sent': chat_completion.choices[0].message.content.strip(), 'triples': chosen_triples}
        with open("./ont_2_music_train.jsonl", "a", encoding='utf-8') as f:
            f.write(json.dumps(res) + "\n")
      