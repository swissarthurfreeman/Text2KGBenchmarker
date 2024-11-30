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
from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint, LearningRateMonitor
import warnings

torch.cuda.empty_cache()
torch.set_float32_matmul_precision('high')

warnings.filterwarnings("ignore", ".*does not have many workers.*")


def train(conf: DictConfig):
    pl.seed_everything(conf.pl_seed, verbose=False)
    
    model_config = AutoConfig.from_pretrained(
        pretrained_model_name_or_path=conf.repo_path + conf.pretrained_model_name_or_path,
        decoder_start_token_id = 0,         # this option is in BartConfig but undocumented, what it does is unclear
        dropout = conf.dropout,             # it has something to do with BART token shifting.
        forced_bos_token_id=None,           # TODO : potentially removable, not in BartConfig, documented parameters will be added as members of config 
        no_repeat_ngram_size=0,             # model.generate() https://huggingface.co/docs/transformers/en/main_classes/text_generation
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
        pretrained_model_name_or_path=conf.repo_path + conf.tokenizer_name_or_path,
        use_fast=True,
        additional_special_tokens=['<obj>', '<subj>', '<triplet>', '<head>', '</head>', '<tail>', '</tail>']
    )
    """Tokenizer of type transformers.models.bart.tokenization_bart_fast.BartTokenizerFast"""
    #print(type(tokenizer))
    
    model: BartForConditionalGeneration = cast(BartForConditionalGeneration, AutoModelForSeq2SeqLM.from_pretrained(
        pretrained_model_name_or_path=conf.repo_path + conf.pretrained_model_name_or_path,
        config=model_config     # forwarded to BartForConditionalGeneration's constructor
    ))
    """transformers.models.bart.modeling_bart.BartForConditionalGeneration"""
    
    model.resize_token_embeddings(len(tokenizer))
    
    pl_data_module = BaseLightningDataModule(conf, tokenizer, model)
    pl_data_module.prepare_data()
    
    pl_module = BaseLightningModule(conf=conf, config=model_config, tokenizer=tokenizer, model=model)
    
    wandb_run_name = get_wandb_run_name(conf)
    # TODO : use ontology name here instead of project
    wandb_logger = WandbLogger(project="wikidata-movies", name=wandb_run_name)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    
    callbacks_list = []
    callbacks_list.append(ModelCheckpoint(
        monitor=conf.monitor_var,                # monitor val_F1_micro
        save_top_k=1,
        verbose=True,
        save_last=True,
        dirpath='wikidata_movies_' + wandb_run_name
    ))
    
    callbacks_list.append(EarlyStopping(
        monitor=conf.monitor_var,               # stop the training if this value doesn't improve
        mode='max',                             
        patience=5                              # for 5 epochs
    ))
    
    callbacks_list.append(LearningRateMonitor(logging_interval='step'))
    
    trainer = pl.Trainer(
        accelerator=device,
        accumulate_grad_batches=conf.gradient_acc_steps,
        gradient_clip_val=conf.gradient_clip_value,
        val_check_interval=conf.val_check_interval,
        max_steps=conf.max_steps,
        precision='16-mixed',
        logger=wandb_logger,
        enable_checkpointing=True,
        callbacks=callbacks_list
    )
    
    trainer.fit(pl_module, datamodule=pl_data_module)
    

def get_wandb_run_name(conf: DictConfig) -> str:
    res = str(conf.num_return_sequences) + "_ret_seq_"
    res += "warm_steps=" + str(conf.warmup_steps) + "_"
    res += "tbs=" + str(conf.train_batch_size) + "_"
    res += "drop=" + str(conf.dropout) + "_"
    
    if conf.relation_mapping:
        res += "rel_map_"
    if conf.sentence_entailement:
        res += "sent_entail_"
        
    return res
    

@hydra.main(config_path="../conf", config_name="root", version_base="1.1")
def main(conf: DictConfig):
    #print("conf")
    #print(omegaconf.OmegaConf.to_yaml(conf))
    train(conf)

if __name__ == '__main__':
    main() 
