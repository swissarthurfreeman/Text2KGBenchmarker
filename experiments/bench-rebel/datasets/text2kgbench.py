import json
import glob
import datasets
from datasets import disable_caching
disable_caching()

class Text2KGBenchConfig(datasets.BuilderConfig):
    def __init__(self, **kwargs):
        super(Text2KGBenchConfig, self).__init__(**kwargs)

class Text2KGBenchData(datasets.GeneratorBasedBuilder):
    # https://huggingface.co/docs/datasets/en/dataset_script   
     
    BUILDER_CONFIGS = [
        Text2KGBenchConfig(
            name = "wikidata_synthetic",
            version = datasets.Version("1.0.0")
        )
    ]
    
    def _info(self):
        return datasets.DatasetInfo(
            description="Synthetic Text2KGBench inspired synthetic dataset",
            features=datasets.Features(
                {
                    "id": datasets.Value("string"),
                    "sent": datasets.Value("string"),
                    "triples": datasets.Value("string")
                }
            )
        )
    
    def _split_generators(self, dl_manager: datasets.DownloadManager | datasets.StreamingDownloadManager) -> list[datasets.SplitGenerator]:
        # BUG : when passing data_files via load_dataset(), the values are put into lists...
        # note, for datasets.SplitGenerator gen_kwargs are arguments to forward to the DatasetBuilder._generate_examples method of the builder.
        # https://huggingface.co/docs/datasets/en/package_reference/builder_classes
        return [
            datasets.SplitGenerator(name = datasets.Split.TRAIN, gen_kwargs={"filepaths": self.config.data_files['train']}),
            datasets.SplitGenerator(name = datasets.Split.VALIDATION, gen_kwargs={"filepaths": self.config.data_files['dev']}),
            datasets.SplitGenerator(name = datasets.Split.TEST, gen_kwargs={"filepaths": self.config.data_files['test']})
        ]
    
    def _generate_examples(self, filepaths: list[str]):
        """return a generator of a linearized text2kgbench examples, when _generate_examples is called, it's code
        isn't run, it just returns a generator which has to be iterated upon, each iteration computes the value on the fly."""
        data: list[dict[str, str | bool | dict[str, str]]] = []
        
        for filepath in filepaths:
            with open(filepath) as json_file:
                # list of dictionaries {id:str, sent: str, verified: bool, unseen:bool, triples:list[{sub:str, rel:str, obj:str}]}
                for line in json_file:
                    json_line = json.loads(line)
                    data.append(json_line)
                
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
            