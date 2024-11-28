import json
import datasets
from datasets import disable_caching
disable_caching()

class SyntheticWikidataConfig(datasets.BuilderConfig):
    def __init__(self, **kwargs):
        super(SyntheticWikidataConfig, self).__init__(**kwargs)

class SyntheticWikidata(datasets.GeneratorBasedBuilder):
    
    BUILDER_CONFIGS = [
        SyntheticWikidataConfig(
            name = "plain_text",
            version = datasets.Version("1.0.0", ""),
            description = "Plain text",
        )
    ]
    
    def _info(self):
        return datasets.DatasetInfo(
            description="Synthetic Text2KGBench inspired synthetic dataset",
            features=datasets.Features(
                {
                    "id": datasets.Value("string"),
                    "sent": datasets.Value("string"),
                    "triples": datasets.Value("string"),
                }
            ),
            supervised_keys=None
        )
    
    def _split_generators(self, dl_manager: datasets.DownloadManager | datasets.StreamingDownloadManager) -> list[datasets.SplitGenerator]:
        return [
            datasets.SplitGenerator(name = datasets.Split.TRAIN, gen_kwargs={"filepath":      "/home/users/f/freemana/Text2KGBenchmarker/data/wikidata_synthetic/train/ont_1_movie_train.jsonl"}),
            datasets.SplitGenerator(name = datasets.Split.VALIDATION, gen_kwargs={"filepath": "/home/users/f/freemana/Text2KGBenchmarker/data/wikidata_synthetic/test/ont_1_movie_test.jsonl"}),
            datasets.SplitGenerator(name = datasets.Split.TEST, gen_kwargs={"filepath":       "/home/users/f/freemana/Text2KGBenchmarker/data/wikidata_synthetic/test/ont_1_movie_test.jsonl"})
        ]
        
    def _generate_examples(self, filepath):
        """return a generator of a linearized text2kgbench examples, when _generate_examples is called, it's code
        isn't run, it just returns a generator which has to be iterated upon, each iteration computes the value on the fly."""
        data: list[dict[str, str | bool | dict[str, str]]] = []
        
        with open(filepath) as json_file:
            print("Open file", filepath, "...")
            # list of dictionaries {id:str, sent: str, verified: bool, unseen:bool, triples:list[{sub:str, rel:str, obj:str}]}
            data = [json.loads(line) for line in json_file]
        
        print("Done opening and parsing", filepath)
        for sent in data:
            lin_triplets: str = ""
            prev_head = None
                
            # sort triples by order of appearance of subject in sentence
            relations_sorted: list[dict] = sorted(sent['triples'], key=lambda s: sent['sent'].find(s['sub']))     
            
            for relation in relations_sorted:
                if prev_head == relation['sub']:            # continuation of triple
                    lin_triplets += f' <subj> ' + relation['obj'] + ' <obj> ' + relation['rel']
                elif prev_head == None:                     # first triple
                    lin_triplets += '<triplet> ' + relation['sub'] + ' <subj> ' + relation['obj'] + ' <obj> ' + relation['rel']
                    prev_head = relation['sub']
                else:                                       # new triple
                    lin_triplets += ' <triplet> ' + relation['sub'] + ' <subj> ' + relation['obj'] + ' <obj> ' + relation['rel']
                    prev_head = relation['sub']
            
            yield sent['id'], {
                'id': sent['id'],
                'sent': sent['sent'],
                'triples': lin_triplets
            }
            