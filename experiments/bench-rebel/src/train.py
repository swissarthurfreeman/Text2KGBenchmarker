import hydra
import omegaconf
import torch
from typing import cast
import pytorch_lightning as pl
from omegaconf import DictConfig
from transformers import AutoConfig, AutoTokenizer, AutoModelForSeq2SeqLM
from transformers.models.bart.configuration_bart import BartConfig
from transformers.models.bart.tokenization_bart_fast import BartTokenizerFast
from transformers.models.bart.modeling_bart import BartForConditionalGeneration
from lightning_data_module import BaseLightningDataModule
from lightning_module import BaseLightningModule
from pytorch_lightning.loggers.wandb import WandbLogger

import warnings

torch.cuda.empty_cache()
torch.set_float32_matmul_precision('high')

warnings.filterwarnings("ignore", ".*does not have many workers.*")


def train(conf: DictConfig):
    pl.seed_everything(conf.pl_seed, verbose=False)
    
    model_config = AutoConfig.from_pretrained(
        pretrained_model_name_or_path=conf.pretrained_model_name_or_path,
        decoder_start_token_id = 0,         # this option is in BartConfig but undocumented, what it does is unclear
        dropout = conf.dropout,             # it has something to do with BART token shifting.
        forced_bos_token_id=None,           # TODO : potentially removable, not in BartConfig, documented parameters will be added as members of config 
        no_repeat_ngram_size=0,             # model.generate() parameter
        early_stopping=False                # model.generate() parameter
    )
    """
    AutoConfig will instantiate the appropriate configuration class for BART/REBEL by inferring it from the model 
    path/name. There's an Auto class for every task, like AutoModelForSeq2SeqLM which will be a seq2seq BART model
    instance. https://huggingface.co/docs/transformers/en/model_doc/auto in this case, it'll be a BartConfig, whose
    description is here, transformers.models.bart.configuration_bart.BartConfig
    https://github.com/huggingface/transformers/blob/main/src/transformers/models/bart/configuration_bart.py#L31
    it'll have a encoder/decoder number of layers/attention heads, ffn_dimensions, standard deviation of layer norm,
    all the different hyperparameters that are needed to define the model.
    Note that the file experiments/rebel/Rebel-large/config.json contains the BartConfig. 
    """
    #print(type(model_config), model_config.no_repeat_ngram_size)
    
    tokenizer = AutoTokenizer.from_pretrained(
        pretrained_model_name_or_path=conf.tokenizer_name_or_path,
        use_fast=True,
        additional_special_tokens=['<obj>', '<subj>', '<triplet>', '<head>', '</head>', '<tail>', '</tail>']
    )
    """Tokenizer of type transformers.models.bart.tokenization_bart_fast.BartTokenizerFast"""
    #print(type(tokenizer))
    
    model: BartForConditionalGeneration = cast(BartForConditionalGeneration, AutoModelForSeq2SeqLM.from_pretrained(
        pretrained_model_name_or_path=conf.pretrained_model_name_or_path,
        config=model_config     # forwarded to BartForConditionalGeneration's constructor
    ))
    """transformers.models.bart.modeling_bart.BartForConditionalGeneration"""
    
    model.resize_token_embeddings(len(tokenizer))
    
    pl_data_module = BaseLightningDataModule(conf, tokenizer, model)
    pl_data_module.prepare_data()
    
    
    pl_module = BaseLightningModule(conf=conf, config=model_config, tokenizer=tokenizer, model=model, 
                                    ontology_path="/home/users/f/freemana/Text2KGBenchmarker/data/wikidata_tekgen/ontologies/ont_1_movie.json", 
                                    wandb_run_name="")
    
    wandb_logger = WandbLogger(project="bench-rebel-rewrite", name="bench-rebel-rewrite")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    trainer = pl.Trainer(
        accelerator=device,
        accumulate_grad_batches=conf.gradient_acc_steps,
        gradient_clip_val=conf.gradient_clip_value,
        val_check_interval=conf.val_check_interval,
        max_steps=conf.max_steps,
        precision='16-mixed',
        enable_checkpointing=False,
        logger=wandb_logger
    )
    
    trainer.fit(pl_module, datamodule=pl_data_module)
    
    
@hydra.main(config_path="../conf", config_name="root", version_base="1.1")
def main(conf: DictConfig):
    #print("conf")
    #print(omegaconf.OmegaConf.to_yaml(conf))
    train(conf)

if __name__ == '__main__':
    main() 
