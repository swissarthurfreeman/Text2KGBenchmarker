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

You can then run the script via `python3 run.py`, to generate, using prompt tuning for ontology guided triple generation with 1 to 6 shots over `wikidata_tekgen` and `dbpedia_webnlg_clean` using the specified model. The responses will be written to `experiments/results/llm_responses/gpt-4o-i-shot` where `i` is the number of training examples provided in the prompt. the file for Wikidata-TekGen's movie ontology GPT-4o responses using 6-shots will be at `experiments/results/llm_responses/gpt-4o-6-shot/ont_1_movie-wikidata_tekgen.jsonl`. 

The querying can be interrupted and re-ran, and the script will pick up from where it left off, note that this takes a couple of hours to deal with the whole dataset, as OpenAI applies API request limitations, the requests cannot be ran in parallel.

**Make sure that the `experiments/results/llm_responses/model_name/` folder doesn't exist, or else the new responses will be appended to the ones already present, if you're generating everything from scratch, the easiest approach is to empty the `experiments/results/llm_responses/` and `experiments/results/metrics/` folders.**

### Generating Metrics

One you have all the response folders generated under `experiments/results/llm_responses/`, you can compute the resulting metrics (Recall, Precision, F1, OC, RH, OH) for every ontology and the global average, in percentile and standard deviation form, using, from within the `experiments/utils/` folder, the script `metrics.py` via `python3 metrics.py`. This will generate a folder for every model under `experiments/results/metrics/model_name/` with a `.jsonl` file containing the *metrics per sample* for every ontology and variant for DBpedia-WebNLG and Wikidata-TekGen in csv and `jsonl` format located in :

- `dbpedia_webnlg_clean_avg.jsonl`
- `dbpedia_webnlg_clean_avg_per_ontology.csv`
- `wikidata_tekgen_avg.jsonl`
- `wikidata_tekgen_avg_per_ontology_all.csv`
- `wikidata_tekgen_avg_per_ontology_unseen.csv`
- `wikidata_tekgen_avg_per_ontology_verified.csv`

as well as global averages, across every ontology, in median and mean form, located in :

- `global_avg.csv`
- `global_median.csv` 


## Using REBEL

The general principle for running an experiment using REBEL is simply to write an appropriate configuration file for the desired experiment placing it at `experiments/bench-rebel/conf/data/config_file.yaml` and running the test or train script overriding the hydra `data` parameter. **Make sure to update the `repo_path` key to the output of `cwd` at the root directory of the repository in the file `experiments/bench-rebel/conf/root.yaml` (we use absolute paths inside REBEL's codebase).** 

[Hydra](https://hydra.cc/docs/intro/) is a python library that allows the specification of structured configuration files in `.yaml` file, it's very useful for machine learning workflows to handle the vast amount of possible hyperparameters of our program. 


### Evaluating Raw REBEL on Test Data

To evaluate REBEL on Text2KGBench, without fine-tuning, using their publicly available checkpoint downloaded under the [Downloading the REBEL Model](#downloading-the-rebel-model) section, we use the `test.py` script under `experiments/bench-rebel/src/test.py`. This script sets up the model and it's tokenizer as well as the lightning data module which is configured in test mode, hence only it's test data loader is configured and passed to a lightning trainer instance in test mode. 

Evaluation is done on the array of test files which must be specified inside the config file via the `test_files` key. The [*dataset script file*](https://huggingface.co/docs/datasets/en/dataset_script) must also be specified which is in charge of parsing the `.jsonl` files of Text2KGBench, we have just one of them, which works for the synthetic, Wikidata-TekGen and DBpedia-WebNLG. 


We provide a configuration file for raw REBEL evaluation on the whole of Text2KGBench's test data inside `experiments/bench-rebel/conf/data/text2kgbench-raw-rebel-test.yaml`. Readers can re-use this configuration by running the test script like so,

```bash
python3 test.py data=text2kgbench-raw-rebel-test
```

Note that this takes about 10 minutes on a GPU such as the NVIDIA RTX A5500, using 3 evaluation beams and 1 return sequence and a batch size of 24 as in the config. The results will be written inside the `experiments/results/llm_responses/rebel-raw-3-beams-2-ret-seq` folder. 
You can then compute the metrics for the model, using the script `metrics.py` under `experiments/utils/metrics.py` from inside that directory.

Once ran, the metrics will be available under `experiments/metrics/rebel-raw-3-beams-2-ret-seq`, the global averages file `global_avg.csv` should look like,
```
dataset,         subset,      P,    R,   F1,   OC,   SH,   RH,   OH
wikidata_tekgen, unseen,   0.14, 0.29, 0.18, 0.47, 0.03, 0.53, 0.04
wikidata_tekgen, verified, 0.18, 0.31, 0.21, 0.45, 0.01, 0.55, 0.02
wikidata_tekgen, all,      0.15, 0.27, 0.18, 0.41, 0.01, 0.59, 0.01
dbpedia_webnlg,  all,      0.07, 0.06, 0.06, 0.29, 0.01, 0.71, 0.01
```

### Fine-Tuning REBEL

In our work, we fine-tuned a seperate REBEL fine tune on every ontology's training data. This is done by using the `train.py` script inside `experiments/bench-rebel/src/` and overriding the appropriate hydra parameters using the `text2kgbench-fine-tune.yaml` config file. 
The parameters that must be overrided are : 
- `wandb_run_name`
- `ontology_paths`
- `train_files`
- `val_files`

#### Fine-Tune on Text2KGBench

A bash script `train_rebel_text2kgbench_raw.sh` is provided, which assumes a SLURM environment (more on this under [Slurm](#slurm) section), if you're not running on Slurm, you can remove the sbatch lines from the two loops inside the script. This script fine-tunes REBEL on only Text2KGBench data, it can be ran via `bash train_wikidata_tekgen.sh`, do not use sh as this file uses bash multi-line syntax. The script submits a fine-tuning job for every ontology inside Text2KGBench, and reports performance metrics in wandb within two seperate projects : `Text2KGBench-Wikidata-TekGen-fine-tune` and `Text2KGBench-DBpedia-WebNLG-fine-tune`. The best checkpoints by validation F1 are saved into the `outputs/date/time/wandb_project_name/wandb_run_name/ontology_name-val_F1_micro=best_f1_reached` where `wandb_run_name` is, for example, `ont_1_movie-Wikidata-TekGen-train-val`.

#### Fine-Tune using Wikidata-Synthetic & Wikidata-TekGen+Synthetic

Another bash script is provided for this, `train_rebel_text2kgbench_synthetic.sh`, which can also be ran via `bash train_rebel_text2kgbench_synthetic.sh`.

The corresponding metrics will be logged within two wandb projects, for each of the techniques : `Wikidata-Synthetic-fine-tune`, `Wikidata-TekGen+Synthetic-fine-tune`.
In both cases, the validation metrics which are plotted are validation on **Text2KGBench's original validation data**, e.g. that of Wikidata-TekGen. 
This is for the sake of getting a performance estimate of the fine-tune on the real data when learning on the synthetic data. 

Since we only have synthetic data for Wikidata-TekGen's ontologies, only 10 fine-tunes are done, for each of the 10 ontologies of that dataset part. 

### Evaluating Fine-Tuned REBEL

The principle for evaluation here is simply to use the checkpoints generated by the fine-tuning of section [Fine-Tune on Text2KGBench](#fine-tune-on-text2kgbench) and evaluate them on Text2KGBench's test data like in section [Using REBEL](#using-rebel). 

Since we're using our own checkpoints, we have to specify the following parameters in the config, 

- `checkpoint_path` the path towards the saved checkpoint
- `output_file_path` the file to the which to write test responses

Since training is stochastic due to dropout, the top validation performance you'll reach will be specific to your run. To evaluate the checkpoints then, you'll need to fill the arrays within the two bash scripts for this :

- `test_rebel_text2kgbench_raw.sh` evaluates the checkpoints trained on raw Text2KGBench data.
- `test_rebel_wikidata-synthetic.sh` evaluates checkpoints trained on Synthetic and Wikidata-TekGen+Synthetic data.


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
