import hydra
import torch
import omegaconf
import pytorch_lightning as pl
from lightning_module import BaseLightningModule
from lightning_data_module import BaseLightningDataModule
from transformers import AutoConfig, AutoTokenizer, AutoModelForSeq2SeqLM


def test(conf: omegaconf.DictConfig):
    """
    Run the model and do inference over one epoch of the test data
    specified by conf.test_file. 
    """
    pl.seed_everything(conf.pl_seed)
    
    model_config = AutoConfig.from_pretrained(
        pretrained_model_name_or_path=conf.repo_path + conf.pretrained_model_name_or_path,
        decoder_start_token_id=0,
        no_repeat_ngram_size=0,
        early_stopping=False
    )
    
    model = AutoModelForSeq2SeqLM.from_pretrained(
        pretrained_model_name_or_path=conf.repo_path + conf.pretrained_model_name_or_path,
        config=model_config
    )
    
    tokenizer = AutoTokenizer.from_pretrained(
        pretrained_model_name_or_path=conf.repo_path + conf.pretrained_model_name_or_path,
        use_fast=True,
        additional_special_tokens=['<obj>', '<subj>', '<triplet>']
    )
    
    model.resize_token_embeddings(len(tokenizer))
    
    conf.do_test_predict = True
    pl_data_module = BaseLightningDataModule(conf, tokenizer, model)
    pl_data_module.prepare_data()
    pl_data_module.setup('test')
    
    pl_module = BaseLightningModule(conf, model_config, tokenizer, model, test_ids=pl_data_module.test_ids)
    
    pl_module.hparams.test_file = pl_data_module.conf.test_file
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    trainer = pl.Trainer(
        accelerator = device
    )
    
    trainer.test(pl_module, dataloaders=pl_data_module.test_dataloader(), ckpt_path=conf.repo_path + conf.checkpoint_path)

@hydra.main(config_path='../conf', config_name='root', version_base="1.1")
def main(conf: omegaconf.DictConfig):
    test(conf)

if __name__ == '__main__':
    main()
    
