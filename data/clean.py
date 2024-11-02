import glob, json, re


def camelCaseToSpaces(word: str) -> str: 
    """Split camel cases to spaces, e.g. 'CamelCaseString Hello hello' -> '  Camel  Case  String   Hello hello'
    all this function does is replace every capital letter 'X' by ' X'."""
    return re.sub('([A-Z][a-z]+)', r' \1', re.sub('([A-Z]+)', r' \1', word))


def clean(word: str) -> str:
    return " ".join(camelCaseToSpaces(word).split()).lower().strip()


train_file_paths = glob.glob("./dpedia_webnlg/test/*.jsonl")

for train_file_path in train_file_paths:
    train_data = []
    with open(train_file_path) as train_f:
        train_data = [json.loads(line) for line in train_f]
        
        for train_line in train_data:
            for triple in train_line["triples"]:
                triple["sub"] = triple["sub"].replace("_", " ")
                triple["rel"] = clean(triple["rel"])
                triple["obj"] = triple["obj"].replace("_", " ")
    
    with open("./dpedia_webnlg_clean/test/" + train_file_path.split("/")[-1], "w") as out_f:
        for train_line in train_data:
            out_f.write(json.dumps(train_line) + "\n")


"""
ont_paths = glob.glob("./dpedia_webnlg/ontologies/*.json")
print(ont_paths)


for ont_path in ont_paths:
    with open(ont_path) as ont_f:
        ont = json.load(ont_f)
        for concept in ont["concepts"]:
            concept["label"] = clean(concept["label"])
            concept["qid"] = clean(concept["qid"])
            
        for relation in ont["relations"]:
            relation["label"] = clean(relation["label"])
            relation["domain"] = clean(relation["domain"])
            relation["range"] = clean(relation["range"])
            relation["pid"] = clean(relation["pid"])
        
        with open("./dpedia_webnlg_clean/ontologies/" + ont_path.split("/")[-1], "w") as out_f:
            out_f.write(json.dumps(ont))
            
"""