import omegaconf        # yaml based configuration support
import hydra            # hierarchical folder based yaml configuration (uses omegaconf)

import pytorch_lightning as pl      # wrapper for pytorch, removes torch boilerplate
from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint

from lightning_data_modules import BasePLDataModule
from lightning_modules import BasePLModule
from transformers import AutoConfig, AutoModelForSeq2SeqLM, AutoTokenizer

from pytorch_lightning.loggers.wandb import WandbLogger         # experiment tracker

from pytorch_lightning.callbacks import LearningRateMonitor
#from generate_samples import GenerateTextSamplesCallback


