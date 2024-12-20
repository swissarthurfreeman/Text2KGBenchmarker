import hydra
import torch
import omegaconf
from typing import cast
from pprint import pprint
from datasets import Dataset
import pytorch_lightning as pl
from omegaconf import DictConfig
from datasets import load_dataset
from datasets import disable_caching
disable_caching()
from torch.utils.data import DataLoader
from transformers import DataCollatorForSeq2Seq
from datasets.formatting.formatting import LazyBatch
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
from transformers.models.bart.configuration_bart import BartConfig
from transformers import AutoConfig, AutoTokenizer, AutoModelForSeq2SeqLM
from transformers.models.bart.tokenization_bart_fast import BartTokenizerFast
from transformers.models.bart.modeling_bart import BartForConditionalGeneration



class BaseLightningDataModule(pl.LightningDataModule):
    """pytorch_lightning data module that prepares data, pre-processes it and dynamically pads it,
    provides test, val and train dataloaders that provide samples {'input_ids': tensor(...), ''}"""
    def __init__(self, conf: DictConfig, tokenizer: AutoTokenizer, model: AutoModelForSeq2SeqLM):
        super().__init__()
        self.conf = conf
        """data yaml configuration, dataset_name, train_file, test_file, val_file..."""
        self.tokenizer: BartTokenizerFast = tokenizer
        """tokenization will be applied in pre-processing to every element in dataset."""
        self.model = model
        """we keep this for the DataCollator because model has a maximum input size constraints."""

        print(conf.train_files)
        self.datasets: dict[str, Dataset] = load_dataset(
            path=conf.repo_path + conf.dataset_script_path, 
            data_files={
                'train': [ conf.repo_path + path for path in conf.train_files ], 
                'dev':   [ conf.repo_path + path for path in conf.val_files ], 
                'test':  [ conf.repo_path + path for path in conf.test_files ]}, 
            trust_remote_code=True
        )
        """A dataset is a directory that contains data files in generic formats (JSON, CSV...) + a 
        dataset script (our .py files) if it requires code to read the data files, 'Dataset' hugging 
        face class is a table. The data that ends up in here is loaded via our dataset scripts.
        
        self.datasets['validation'].batch(B) will yield a dictionary of {'id': ['id1', 'id2', ...], 'triplets': ["<triplet> ..."]]}
        where lists at keys 'id' and 'triplets' are of length B, the batch size, and key 'triplets'[k] will contain a single linearized
        triplet key for the entry with id at key 'id'[k]
        """
    
        self.text_key   = conf.text_key
        """key of the jsonl entries that contain the full sentence."""
        self.target_key = conf.target_key
        """key of the jsonl entries that contains the triples list."""
        
        self.data_collator = DataCollatorForSeq2Seq(self.tokenizer, self.model)
        """forms a batch from a list of dataset elements by applying some processing, like padding or masking,
        to collate means to collect and combine, this collator will dynamically pad the inputs and labels received,
        this allows making sure all sentences in the batch have a same length, for matrix multiplication and model
        maximum input length compatibility."""
        
        self.column_names = self.datasets['train'].column_names
        
    def prepare_data(self) -> None:
        if self.conf.do_test_predict:
            self.test_ids: list[str] = self.datasets["test"]["id"]
            
            self.test_dataset = self.datasets["test"].map(
                self.preprocess_function,
                batched=True,
                batch_size=100,
                remove_columns=self.column_names,
                #load_from_cache_file=True,                          # under keys 'input_ids' and 'labels', but no more original data
                #cache_file_name=self.conf.test_file.replace('jsonl', '-') + self.conf.dataset_script_path.split("/")[-1].replace('.py', '.cache')
            )
        else:
            self.train_dataset = self.datasets['train'].map(
                self.preprocess_function,                           # apply this function to every sample
                batched=True,                                       # in batch mode, provides speedup
                batch_size=100,
                remove_columns=self.column_names,                    # output will contains tokenized sentence and triples
                #load_from_cache_file=True,                          # under keys 'input_ids' and 'labels', but no more original data
                #cache_file_name=self.conf.train_file.replace('jsonl', '-') + self.conf.dataset_script_path.split("/")[-1].replace('.py', '.cache')
            )
            
            self.eval_dataset = self.datasets["validation"].map(
                self.preprocess_function,
                batched=True,
                batch_size=100,
                remove_columns=self.column_names,
                #load_from_cache_file=True,                          # under keys 'input_ids' and 'labels', but no more original data
                #cache_file_name=self.conf.val_file.replace('jsonl', '-') + self.conf.dataset_script_path.split("/")[-1].replace('.py', '.cache')
            )
        
        
    def preprocess_function(self, batch: dict[str, list[str]]) -> dict[str, torch.Tensor]:
        """function to apply to every element of dataset, an element is {id: [...], sent: [...], triples:[...]},
        returns a sample with {'id': [...], 'input_ids': [...], 'labels': [...]}"""
        
        inputs = batch[self.text_key]          # batch size of 1000, this is a list of sentences
        targets = batch[self.target_key]       # this is the list of it's linearized triples
        
        #print("Hello", len(inputs), inputs[0], targets[0], "\n\n")
        #exit(0)
        # see parameters https://huggingface.co/docs/transformers/en/main_classes/tokenizer
        # padding is done by collator, not the tokenizer, truncation from 1024 (tokenized) tokens.
        # it'll literally just trunkate the resulting input_ids and attention_mask arrays to 1024 in length
        model_inputs = self.tokenizer(inputs, max_length=1024, padding=False, truncation=True)
        
        labels = self.tokenizer(text_target=targets, max_length=1024, padding=False, truncation=True)
            
        model_inputs["labels"] = labels["input_ids"]
        
        # this will print the list of tokens of the first sentence of this batch, and it's target linearized triples
        # see https://github.com/huggingface/transformers/issues/22306
        #print(self.tokenizer.convert_ids_to_tokens(model_inputs["input_ids"][0]), 
        #      self.tokenizer.convert_ids_to_tokens(model_inputs["labels"][0]))
        #exit(0)
        return model_inputs
    
    def train_dataloader(self) -> DataLoader:
        return DataLoader(
            dataset=self.train_dataset,
            batch_size=self.conf.train_batch_size,
            collate_fn=self.data_collator,
            pin_memory=True
        )
        
    def val_dataloader(self) -> DataLoader:
        return DataLoader(
            dataset=self.eval_dataset,
            batch_size=self.conf.val_batch_size,
            collate_fn=self.data_collator,
            pin_memory=True
        )
        
    def test_dataloader(self) -> DataLoader:
        return DataLoader(
            dataset=self.test_dataset,
            batch_size=self.conf.test_batch_size,
            collate_fn=self.data_collator,
            pin_memory=True
        )
        

if __name__ == '__main__':
    """Simple tests for the sake of comprehension"""
    @hydra.main(config_path="../conf", config_name="root", version_base="1.1")
    def main(conf: DictConfig):
        model_config = AutoConfig.from_pretrained(
            conf.repo_path + conf.pretrained_model_name_or_path, decoder_start_token_id = 0, dropout = conf.dropout, 
            forced_bos_token_id=None, no_repeat_ngram_size=0, early_stopping=False                
        )
        
        # additional_special_tokens adds the provided tokens to the vocabulary
        tokenizer = AutoTokenizer.from_pretrained(
            pretrained_model_name_or_path=conf.repo_path + conf.tokenizer_name_or_path, use_fast=True,
            additional_special_tokens=['<obj>', '<subj>', '<triplet>', '<head>', '</head>', '<tail>', '</tail>']
        )
        
        model: BartForConditionalGeneration = cast(BartForConditionalGeneration, AutoModelForSeq2SeqLM.from_pretrained(
            pretrained_model_name_or_path=conf.repo_path + conf.pretrained_model_name_or_path,
            config=model_config     # forwarded to BartForConditionalGeneration's constructor
        ))
        
        pl_data_module: BaseLightningDataModule = BaseLightningDataModule(conf, tokenizer, model)
        pl_data_module.prepare_data()
        
        val_dataloader = pl_data_module.val_dataloader()
        
        for i in range(10):
            print(tokenizer.convert_ids_to_tokens(i))
        
        for batch in iter(val_dataloader):
            print(type(batch), batch.keys(), len(batch['input_ids']), type(batch['input_ids']), len(batch['input_ids'][1]))
            print(tokenizer.convert_ids_to_tokens(batch['input_ids'][0]))
            break

    main()