import torch
from datasets import Dataset
import pytorch_lightning as pl
from omegaconf import DictConfig
from datasets import load_dataset
from transformers import DataCollatorForSeq2Seq
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer, Seq2SeqLMOutput
from torch.nn import CrossEntropyLoss

class BaseLightningDataModule(pl.LightningDataModule):
    
    def __init__(self, conf: DictConfig, tokenizer: AutoTokenizer, model: AutoModelForSeq2SeqLM):
        super().__init__()
        self.conf = conf
        """data yaml configuration, dataset_name, train_file, test_file, validation_file..."""
        self.tokenizer = tokenizer
        """tokenization will be applied in pre-processing to every element in dataset."""
        self.model = model
        """we keep this for the DataCollator because model has a maximum input size."""
        
        data_files_paths={
            'train': conf.train_file, 
            'dev': conf.validation_file, 
            'test': conf.test_file
        }
        
        self.datasets: dict[str, Dataset] = load_dataset(
            conf.dataset_name, 
            data_files=data_files_paths, 
            trust_remote_code=True
        )
        """A dataset is a directory that contains data files in generic formats (JSON, CSV...) + a 
        dataset script (our .py files) if it requires code to read the data files, Dataset hugging 
        face class is a table."""
    
        self.text_key   = conf.text_key
        """key of the jsonl entries that contain the full sentence."""
        self.target_key = conf.target_key
        """key of the jsonl entries that contains the triples list."""
        
        self.data_collator = DataCollatorForSeq2Seq(self.tokenizer, self.model)
        """forms a batch from a list of dataset elements by applying some processing, like padding or masking,
        to collate means to collect and combine, this collator will dynamically pad the inputs and labels received,
        this allows making sure all sentences in the batch have a same length, for matrix multiplication and model
        maximum input length compatibility."""
        
    def prepare_data(self) -> None:
        self.train_dataset = self.datasets['train'].map(
            self.preprocess_function,                           # apply this function to every sample
            batched=True,                                       # in batch mode, provides speedup
            remove_columns=self.datasets['train'].column_names, # output will contains tokenized sentence and triples
            load_from_cache_file=False                          # under keys 'input_ids' and 'labels', but no more original data
        )
        
        self.eval_dataset = self.datasets["validation"].map(
            self.preprocess_function,
            batched=True,
            remove_columns=self.datasets["validation"].column_names,
            load_from_cache_file=False
        )
        
    def preprocess_function(self, sample: dict[str, str | list[dict[str, str]]]) -> dict[str, torch.Tensor]:
        """function to apply to every element of dataset, an element is {id:..., sent:..., triples:[...]}"""
        inputs = sample[self.text_key]
        targets = sample[self.target_key]
        
        # see parameters https://huggingface.co/docs/transformers/en/main_classes/tokenizer
        # padding is done by collator, not the tokenizer, truncation from 1024 (tokenized) tokens.
        model_inputs = self.tokenizer(inputs, max_length=1024, padding=False, truncation=True)
        
        with self.tokenizer.as_target_tokenizer():
            labels = self.tokenizer(targets, max_length=1024, padding=False, truncation=True)
            
        model_inputs["labels"] = labels["input_ids"]
        return model_inputs