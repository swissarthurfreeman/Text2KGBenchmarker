import re
import json

# replace this film, the film, the series by the first subject

references = [
    "This Film",
    "The film",
    "this film",
    "the film",
    "the series",
    "The series",
    "The episode",
    "the episode"
]

def replace(sentence: str, subject: str):    
    for ref in references:
        sentence = sentence.replace(ref, subject)
    return sentence

def clean(sent: str):
    return "".join(sent.lower().split()).strip()

with open("./ont_1_movie_train.jsonl", "r") as f:
    data = [json.loads(line) for line in f]
    for sent in data:
        
        
        for fact in sent["triples"]:
            if fact["rel"] == "publication date":
                fact["obj"] = "01 January " + re.search("[0-9]{4}", fact["obj"]).group()
            
            if clean(sent["sent"]).find(clean(sent["triples"][0]["sub"])) == -1:
                sent["sent"] = replace(sent["sent"], sent["triples"][0]["sub"])
    
    with open("./ont_1_movie_train_clean.jsonl", "a") as g:
        for sent in data:
            g.write(json.dumps(sent) + "\n")
            
