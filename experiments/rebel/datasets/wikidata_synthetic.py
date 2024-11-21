import json
import datasets

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
                    "title": datasets.Value("string"),
                    "context": datasets.Value("string"),
                    "triplets": datasets.Value("string"),
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
        """generate linearized text2kgbench examples"""
        data: list[dict[str, str | bool | dict[str, str]]] = []
        with open(filepath) as json_file:
            # list of dictionaries {id:str, sent: str, verified: bool, unseen:bool, triples:list[{sub:str, rel:str, obj:str}]}
            data = [json.loads(line) for line in json_file]
        
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
            
            #print("\n\n#################### text2kgbench datasetGenerator output \n", "sent :", sent['sent'], "\n triples :", sent['triples'], "\n linearized :", lin_triplets)
            #print("\n\n")
            yield sent['id'], {
                'title': sent['id'],
                'id': sent['id'],
                'context': sent['sent'],
                'triplets': lin_triplets
            }
            