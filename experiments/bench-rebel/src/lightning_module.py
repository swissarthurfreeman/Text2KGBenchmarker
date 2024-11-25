import torch
import pytorch_lightning as pl
from transformers import AutoConfig, AutoTokenizer, AutoModelForSeq2SeqLM

class BaseLightningModule(pl.LightningModule):
    def __init__(self, conf, config: AutoConfig, tokenizer: AutoTokenizer, model: AutoModelForSeq2SeqLM, wandb_run_name: str, *args, **kwargs):
        """
        Parameters
        ----------
        - conf the hydra configuration contained in the conf folder
        - config the facebook/BART/REBEL configuration
        - tokenizer the BART/REBEL tokenizer
        - model the BART/REBEL model
        - wandb_run_name, used to log the different training metrics
        """
        
        self.save_hyperparameters(conf)     # adds conf dict to self.hparams
        self.tokenizer = tokenizer
        self.model = model
        self.config = config
        self.wandb_run_name = wandb_run_name
        
        self.loss_fn = torch.nn.CrossEntropyLoss(ignore_index=-100)
        
        self.val_preds = [] # todo : add typing indications to this
        self.test_preds = []
        
         
        
    