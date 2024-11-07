import json
import datasets

class Text2KGBenchWikidataConfig(datasets.BuilderConfig):
    def __init__(self, **kwargs):
        super(Text2KGBenchWikidataConfig, self).__init__(**kwargs)

class Text2KGBenchWikidata(datasets.GeneratorBasedBuilder):
    
    def _split_generators(self, dl_manager: datasets.DownloadManager | datasets.StreamingDownloadManager) -> list[datasets.SplitGenerator]:
        return [
            datasets.SplitGenerator(name = datasets.Split.TEST, gen_kwargs={"filepath": "/home/gordon/Documents/edu/gordon_ms/Project/Benchmarker/data/wikidata_tekgen/test/ont_1_movie_test.jsonl"}),
            datasets.SplitGenerator(name = datasets.Split.TRAIN, gen_kwargs={"filepath": "/home/gordon/Documents/edu/gordon_ms/Project/Benchmarker/data/wikidata_tekgen/train/ont_1_movie_train.jsonl"}),
            datasets.SplitGenerator(name = datasets.Split.VALIDATION, gen_kwargs={"filepath": "/home/gordon/Documents/edu/gordon_ms/Project/Benchmarker/data/wikidata_tekgen/validation/ont_1_movie_validation.jsonl"})
        ]
        
    def _generate_examples(self, filepath):
        """generate linearized text2kgbench examples on movie ontology"""
        with open(filepath) as json_file:
            # list of dictionaries {id:str, sent: str, verified: bool, unseen:bool, triples:list[{sub:str, rel:str, obj:str}]}
            data: list[dict[str, str | bool | dict[str, str]]] = [json.loads(line) for line in json_file]
            
            for sent in data:
                lin_triplets: str = ""
                prev_head = None
                # NOTE : we drop the triple when subject isn't in the sentence, e.g. "The series was directed by x", ["series name", "directed by", "x"]
                for triple in sent["triples"]:
                    if sent["sent"].find(triple["sub"]) == -1:
                        continue
                    
                # sort triples by order of appearance of subject in sentence
                relations_sorted: list[dict] = sorted(sent['triplets'], key=lambda s: sent['sent'].find(s['sub']))     
                
                for relation in relations_sorted:
                    if prev_head == relation['sub']:            # continuation of triple
                        lin_triplets += f' <sub> ' + relation['obj'] + ' <obj> ' + relation['rel']
                    elif prev_head == None:                     # first triple
                        lin_triplets += '<triplet> ' + relation['sub'] + ' <sub> ' + relation['obj'] + ' <obj> ' + relation['rel']
                        prev_head = relation['sub']
                    else:                                       # new triple
                        lin_triplets += ' <triplet> ' + relation['sub'] + ' <sub> ' + relation['obj'] + ' <obj> ' + relation['rel']
                        prev_head = relation['sub']
                
                yield sent['id'], {
                    'title': sent['id'],
                    'id': sent['id'],
                    'context': sent['sent'],
                    'triplets': lin_triplets
                }
            