# Text2KGBenchmarker : Master Thesis Repository of A. Freeman

## Architecture

This repository consists in the following main parts : 

- A cleaned and simplified version of the [Text2KGBench dataset](https://github.com/cenguix/Text2KGBench) located under `data/dbpedia_webnlg_clean` and `data/wikidata_tekgen`. The precise changes done to the original Text2KGBench dataset published alongside the [2023 paper](https://arxiv.org/abs/2308.02357) are detailed under `data/CHANGES.md`.  
- A simplified, documented version of the [REBEL model repository](https://github.com/Babelscape/rebel) under `experiments/bench-rebel`, tailored for fine-tuning on Text2KGBench, stripped of any code not fitting our study's use cases. 
- A suite of utility scripts located under `experiments/utils`, responding to various use-cases such as metrics, graphics, normalizations (relational mapping, sentence entailement) and prompt tuning generation tasks. 
- An `experiments/results` folder, containing all model variation answers for Text2KGBench, such as `experiments/results/Babelscape.rebel-large-6-beams-rel-map/` for 6 return sequences with relational mapping REBEL model directly evaluated on Text2KGBench's test data, where the folder contains a `.jsonl` file for every test ontology samples file. 
- A synthetic dataset under `data/wikidata_synthetic`, generated using Wikidata and GPT-4o with the same ontologies as in `data/wikidata_tekgen/ontologies`.

## Installing the Environment

**Assuming a clean installation of Linux** (these commands were tested in an Ubuntu 24.04.1 LTS virtual machine), you can run the following commands to install all required dependencies.

```sudo apt update && sudo apt upgrade```

```sudo apt install git && sudo apt install python3-pip```

```sudo apt install pipx && pipx ensurepath```

Relaunch your terminal, then run,

```pipx install pipenv```

Relaunch your terminal again, then clone the repository, this takes a while, there's 500 MB of data in the repository.

```git clone https://github.com/swissarthurfreeman/Text2KGBenchmarker.git && cd Text2KGbenchmarker```

Finally, install all pipfile dependencies via, 

```pipenv install --verbose```

This takes a while too, pytorch, huggingface, etc must be downloaded, the `--verbose` argument will detail what is being downloaded, it'll certainly take some time with pytorch, which is 1GB large. 


Once this is done, launch a shell via,

```pipenv shell```

You are now inside a pipenv virtual environment with all dependencies for this project. You should be able to run python3 and import any of the `Pipfile` dependencies.

```bash
vboxuser@virtual-machine:~/Text2KGbenchmarker$ pipenv shell
Launching subshell in virtual environment...
vboxuser@virtual-machine:~/Text2KGbenchmarker$ source /home/vboxuser/.local/share/virtualenvs/Text2KGbenchmarker-ntCgD4G7/bin/activate
(Text2KGbenchmarker) vboxuser@virtual-machine:~/Text2KGbenchmarker$ python3
Python3 3.12.3 (main, Jan 17 2025, 18:03:48) [GCC 13.3.0] on linux
Type "help", "copyright", "credits" or "license" for more information.
>>> import torch
>>> import SPARQLWrapper	# Works without issues.
```

Note that the path provided above points towards the python interpeter that should be selected in visual studio code, for correct import resolution. 

```/home/vboxuser/.local/share/virtualenvs/Text2KGbenchmarker-ntCgD4G7/bin/python```

Once this is done, you can successfully run any of the scripts of the repository.

### Downloading the REBEL Model

REBEL must be downloaded and installed within the `experiments/Rebel-large/` folder. 
You can downloaded it from [this link](https://osf.io/rxmze?view_only=87e7af84c0564bd1b3eadff23e4b7e54) as provided in the original REBEL repository's instructions. The zip file should then be extracted and all contents should be placed within `experiments/Rebel-large`, such
as the hierarchy contains,

```
Text2KGBenchmarker
|
└───experiments
    │   
    │
    └───bench-rebel
		|
		└───Rebel-large
				added_tokens.json
				config.json
				merges.txt
				special_tokens_map.json
				tokenizer_config.json
				vocab.json
```

Note that the zip file is 1.4GB large, so a decent connection is required. It can be dropped via `ssh` into Baobab using drag and drop.

## Running Experiments 

### Evaluating Prompt Tuning

To reproduce our results for prompt tuning using GPT-4o/GPT-3.5-Turbo, you need an [OpenAI API key](https://platform.openai.com/api-keys).
To this end, you need to create an OpenAPI platform account and credit your account. 

![alt text](image.png)

An example (deactivated) key could be, 
```
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
```

This key should be included in the file `experiments/utils/run.py` inside the `OpenAIAdapter()` constructor at line 92.

```python
model_adapter = OpenAIAdapter(
        "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", 
        "gpt-4o"
)
```

Note that the second argument specifies the OpenAI model to use, if just `gpt-4o`, it'll use the latest version of GPT-4o available. To reproduce our exact results, users should use the checkpoint we used at the time of running our experiments i.e. `gpt-4o-2024-11-20`.
You can also use `gpt-3.5-turbo` to reproduce it's results. 

You can then run the script via `python3 run.py`, to generate, using prompt tuning with 1 to 6 shots over `wikidata_tekgen` and `dbpedia_webnlg_clean` using GPT-4o the responses for ontology guided triple generation. The responses will be written to `experiments/results/llm_responses/gpt-4o-i-shot` where `i` is the number of training examples provided in the prompt. the file for Wikidata-TekGen's movie ontology GPT-4o responses using 6-shots will be at `experiments/results/llm_responses/gpt-4o-6-shot/ont_1_movie-wikidata_tekgen.jsonl`. 

The querying can be interrupted and re-ran, and the script will pick up from where it left off.


**Make sure that the `experiments/results/llm_responses/model_name/` folder doesn't exist, or else the new responses will be appended to the ones already present, if you're generating everything from scratch, the easiest approach is to empty the `experiments/results/llm_responses/` and `experiments/results/metrics/` folders.**

### Generating Metrics

One you have all the response folders generated under `experiments/results/llm_responses/`, you can compute the resulting metrics (Recall, Precision, F1, OC, RH, OH) for every ontology and the global average, in percentile and standard deviation form, using, from the `experiments/utils/` folder, the script `metrics.py` via `python3 metrics.py`. This will generate a folder for every model under `experiments/results/metrics/model_name/` with a `.jsonl` file containing the *metrics per sample* for every ontology and variant for DBpedia-WebNLG and Wikidata-TekGen in csv and `jsonl` format located in :

- `dpedia_webnlg_clean_avg.jsonl`
- `dbpedia_webnlg_clean_avg_per_ontology.csv`
- `wikidata_tekgen_avg.jsonl`
- `wikidata_tekgen_avg_per_ontology_all.csv`
- `wikidata_tekgen_avg_per_ontology_unseen.csv`
- `wikidata_tekgen_avg_per_ontology_verified.csv`

as well as global averages, across every ontology, in median and mean form, located in :

- `global_avg.csv`
- `global_median.csv` 


## Using REBEL

The general principle for running an experiment using REBEL is simply to write an appropriate configuration file for the desired experiment placing it at `experiments/bench-rebel/conf/data/config_file.yaml` and running the test or train script overriding the hydra `data` parameter. Make sure to provide a `dataset_script_path` located in your `config_file.yaml` and to update the `repo_path` key to the output of `cwd` at the root directory of the repository in the file `experiments/bench-rebel/conf/root.yaml` (we use absolute paths inside REBEL's codebase). 

[Hydra](https://hydra.cc/docs/intro/) is a python library that allows the specification of structured configuration files in `.yaml` file, it's very useful for machine learning workflows to handle the vast amount of possible hyperparameters of our program. 


### Evaluating Raw REBEL on Test Data

To evaluate REBEL on Text2KGBench, without fine-tuning, using their publicly available checkpoint downloaded under the [Downloading the REBEL Model](#downloading-the-rebel-model) section, we use the `test.py` script under `experiments/bench-rebel/src/test.py`. This script sets up the model and it's tokenizer as well as the lightning data module which is configured in test mode, hence only it's test data loader is configured and passed to a lightning trainer instance in test mode. 

To evaluate, an array of test files must be specified inside the config file. This is done via the `test_files` key, the [*dataset script file*](https://huggingface.co/docs/datasets/en/dataset_script) must also be specified. This is the file in charge of reading the `.jsonl` files of Text2KGBench, we have just one of them, which works for the synthetic, Wikidata-TekGen or DBpedia-WebNLG by reading the files list from `test_files`. The script is under `experiments/bench-rebel/datasets/text2kgbench.py`. 

```yaml


```


### Fine-Tuning REBEL

### Evaluating Fine-Tuned REBEL

## Slurm

If you're running inside a Slurm environment, such as that of the University of Geneva's Baobab cluster, you'll have to use the Slurm CLI to request appropriate resources. To run REBEL, you need a GPU with at least 24GB of Vram. You connect to baobab using,

```$ ssh isis_username@login1.baobab.hpc.unige.ch```

You can view your list of running or pending jobs using, 

```$ squeue -u isis_username```

You can request an interactive terminal with a GPU attached using, 

```$ salloc --ntasks 1 --mem=25G --time=2:00:00 --partition=shared-gpu --gres=gpu:1,VramPerGpu:24G```

Note that there are two parameters for memory, `--mem` requests RAM, which must be specified, or else by default only 2GB are allocated, which will yield an out of memory when instantiating the data loaders. `--gres=gpu:1,VramPerGpu:24G` allows requesting a GPU with a minimum of 24GB of Vram. They are limited, so this can take some time, during weekends and vacations access is usually instantaneous. You can check wether sufficient VRAM was correctly allocated using `nvidia-smi` on the CLI. 

Once you have the allocation, assuming you've installed your pipenv environment before hand, you can activate the usage of python via the following commands,

```$ module load GCCcore/13.2.0 Python/3.11.5 && pipenv shell```

once inside the pipenv shell, you should have access to all pipenv installed dependencies, and should be able to import pytorch and move a tensor to the GPU. One example of shell output could be the following, 

```shell
(baobab)-[isis_username@gpu020 Text2KGBenchmarker]$ module load GCCcore/13.2.0 Python/3.11.5 && pipenv shell
Launching subshell in virtual environment...
 source /home/users/f/isis_username/.local/share/virtualenvs/Text2KGBenchmarker-yg4X5boN/bin/activate
(baobab)-[isis_username@gpu020 Text2KGBenchmarker]$  source /home/users/f/isis_username/.local/share/virtualenvs/Text2KGBenchmarker-yg4X5boN/bin/activate
(Text2KGBenchmarker) (baobab)-[isis_username@gpu020 Text2KGBenchmarker]$ python3
iPython 3.11.5 (main, Nov 12 2024, 14:17:18) [GCC 13.2.0] on linux
Type "help", "copyright", "credits" or "license" for more information.
>>> import torch
>>> torch.ones((1, 10)).to('cuda')
tensor([[1., 1., 1., 1., 1., 1., 1., 1., 1., 1.]], device='cuda:0')
>>>
```

## Fine-Tuning on Text2KGBench

```
srun --gpus=1 --mem-per-gpu=32G --partition=shared-gpu --time=1:00:00 python3 train.py model=rebel_model data=text2kgbench_data train=text2kgbench_train
```

```
python3 test.py model=rebel_model data=nyt_data train=nyt_train do_predict=True checkpoint_path='/home/users/f/isis_username/Text2KGBenchmarker/experiments/rebel/src/outputs/2024-11-05/12-06-55/experiments/nyt/epoch\=8-step\=21078.ckpt'
```

### Fine Tune on Wikidata Movie Ontology

```
python3 train.py model=rebel_model data=text2kgbench_data train=text2kgbench_train
```

Possible output is, 

```
        ALL      TP: 7402;      FP: 751;        FN: 711
                (m avg): precision: 90.79;      recall: 91.24;  f1: 91.01 (micro)
                (M avg): precision: 79.41;      recall: 78.35;  f1: 78.77 (Macro)

        country of citizenship:         TP: 511;        FP: 82; FN: 37; precision: 86.17;       recall: 93.25;  f1: 89.57;      593
        headquarters location:  TP: 17; FP: 1;  FN: 0;  precision: 94.44;       recall: 100.00; f1: 97.14;      18
        contains administrative territorial entity:     TP: 501;        FP: 13; FN: 25; precision: 97.47;       recall: 95.25;  f1: 96.35;      514
        shareholders:   TP: 30; FP: 1;  FN: 2;  precision: 96.77;       recall: 93.75;  f1: 95.24;      31
        country of origin:      TP: 1;  FP: 0;  FN: 0;  precision: 100.00;      recall: 100.00; f1: 100.00;     1
        denonym:        TP: 0;  FP: 0;  FN: 1;  precision: 0.00;        recall: 0.00;   f1: 0.00;       0
        major shareholder:      TP: 31; FP: 1;  FN: 1;  precision: 96.88;       recall: 96.88;  f1: 96.88;      32
        location:       TP: 3570;       FP: 320;        FN: 261;        precision: 91.77;       recall: 93.19;  f1: 92.48;      3890
        founded by:     TP: 46; FP: 13; FN: 15; precision: 77.97;       recall: 75.41;  f1: 76.67;      59
        employer:       TP: 373;        FP: 53; FN: 44; precision: 87.56;       recall: 89.45;  f1: 88.49;      426
        advisors:       TP: 3;  FP: 0;  FN: 0;  precision: 100.00;      recall: 100.00; f1: 100.00;     3
        place of death:         TP: 88; FP: 27; FN: 43; precision: 76.52;       recall: 67.18;  f1: 71.54;      115
        industry:       TP: 0;  FP: 0;  FN: 0;  precision: 0.00;        recall: 0.00;   f1: 0.00;       0
        ethnicity:      TP: 1;  FP: 0;  FN: 0;  precision: 100.00;      recall: 100.00; f1: 100.00;     1
        place of birth:         TP: 170;        FP: 87; FN: 90; precision: 66.15;       recall: 65.38;  f1: 65.76;      257
        country:        TP: 507;        FP: 18; FN: 17; precision: 96.57;       recall: 96.76;  f1: 96.66;      525
        residence:      TP: 479;        FP: 95; FN: 118;        precision: 83.45;       recall: 80.23;  f1: 81.81;      574
        member of sports team:  TP: 17; FP: 1;  FN: 0;  precision: 94.44;       recall: 100.00; f1: 97.14;      18
        child:  TP: 33; FP: 7;  FN: 7;  precision: 82.50;       recall: 82.50;  f1: 82.50;      40
        religion:       TP: 5;  FP: 0;  FN: 0;  precision: 100.00;      recall: 100.00; f1: 100.00;     5
        neighborhood of:        TP: 345;        FP: 21; FN: 29; precision: 94.26;       recall: 92.25;  f1: 93.24;      366
        capital:        TP: 653;        FP: 7;  FN: 7;  precision: 98.94;       recall: 98.94;  f1: 98.94;      660
        location of formation:  TP: 21; FP: 4;  FN: 14; precision: 84.00;       recall: 60.00;  f1: 70.00;      25
        occupation:     TP: 0;  FP: 0;  FN: 0;  precision: 0.00;        recall: 0.00;   f1: 0.00;       0
Testing DataLoader 0: 100%|█████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 313/313 [05:10<00:00,  1.01it/s]
┏━━━━━━━━━━━━━━━━━━━━━━━━━━━┳━━━━━━━━━━━━━━━━━━━━━━━━━━━┓
┃        Test metric        ┃       DataLoader 0        ┃
┡━━━━━━━━━━━━━━━━━━━━━━━━━━━╇━━━━━━━━━━━━━━━━━━━━━━━━━━━┩
│       test_F1_micro       │     91.01192474365234     │
│         test_loss         │    0.07760083675384521    │
│      test_prec_micro      │     90.78866577148438     │
│     test_recall_micro     │     91.23628997802734     │
└───────────────────────────┴───────────────────────────┘
```


### Hydra

Idea is to compose configuration across a folder and `.yaml` hierarchy of files.
See the [intro](https://hydra.cc/docs/intro/).


### Relational Mapping Issues

>>> from sentence_transformers import SentenceTransformer
>>> sent_embedder = SentenceTransformer('sentence-transformers/all-mpnet-base-v2', device="cpu")
>>> llm_triples = ["Hunt screenwriter Michael", "Hunt director Michael"]
>>> ont_relations = ["film director human", "film cost human"]
>>> llm_triples_embed = sent_embedder.encode(llm_triples)
>>> ont_relations_embed = sent_embedder.encode(ont_relations)
>>> sent_embedder.similarity(llm_triples_embed, ont_relations_embed)
tensor([[0.4243, 0.2038],
        [0.4454, 0.2597]])
>>> sent_embedder.similarity(llm_triples_embed[1], ont_relations_embed[0])
tensor([[0.4454]])
>>> sent_embedder.similarity(llm_triples_embed[0], ont_relations_embed[1])
tensor([[0.2038]])
>>> sent_embedder.similarity(llm_triples_embed[0], ont_relations_embed[0])
tensor([[0.4243]])
>>> 

Not sensitive enough, it seems sentence embeddings work better on longer sentences,

```python
from sentence_transformers import SentenceTransformer
sent_embedder = SentenceTransformer('sentence-transformers/all-mpnet-base-v2', device="cpu")
def sent_similarity(sent1, sent2): 
    return sent_embedder.similarity(sent_embedder.encode(sent1), sent_embedder.encode(sent2))

>>> sent_similarity("film screenwriter human", "the film Hunt has as cast member Michael Bay")
tensor([[0.2190]])
>>> sent_similarity("film screenwriter human", "the film Hunt has as screenwriter Michael Bay")
tensor([[0.3998]])
```