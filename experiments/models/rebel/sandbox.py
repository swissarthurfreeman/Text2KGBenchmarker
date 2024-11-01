import hydra
from omegaconf import DictConfig, OmegaConf

"""
import omegaconf        # yaml based configuration support
import hydra            # hierarchical folder based yaml configuration (uses omegaconf)

import pytorch_lightning as pl      # wrapper for pytorch, removes torch boilerplate
from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint

from pl_data_modules import BasePLDataModule
from pl_modules import BasePLModule
from transformers import AutoConfig, AutoModelForSeq2SeqLM, AutoTokenizer

AutoConfig.from_pretrained()

from pytorch_lightning.loggers.neptune import NeptuneLogger
from pytorch_lightning.loggers.wandb import WandbLogger

from pytorch_lightning.callbacks import LearningRateMonitor
from generate_samples import GenerateTextSamplesCallback
"""


@hydra.main(version_base=None, config_path="conf", config_name="root")
def my_app(cfg : DictConfig) -> None:
    print(OmegaConf.to_yaml(cfg))
    
    print(cfg.train.prop1)

if __name__ == "__main__":
    my_app()
    
