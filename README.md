## Benchmarker : Text2KGBench

This is a Text2KGBench cleaned dataset with a rerwite of utility functions to fix technical debt
from the original repository to make working with the dataset easier. 

The list of changes is as follows : 
- data is all kept in `jsonl` files in the hierarchy below, instead of a mix of data across multiple `jsonl` and `txt` files.
- ontology files were edited to remove ghost wikidata IDs which did not a corresponding property or entity label within the file.
- addition of missing domain and range classes to certain properties, to avoid for instance, describing a publication date via `publication_date(film,)`, instead we add `date` as range to yield `publication_date(film, date)`.
- folded all duplicate sentences in `wikidata_tekgen` test and train datasets, aggregating into the same list the facts from train. Indeed, in the train files, certain objects with different ids had the same sentences but with different facts, they were folded into a single id and their facts were concatenated into a list following the same format as `dbpedia_webnlg`. The same processing was done on the `test` data of `wikidata_tekgen` where additionally the `similars` list was updated with the new id folds, replacing the removed sentences by the id of the sentence folded to.  
- `wikidata_tekgen` dataset still has many problems. There are a lot of triples missing, for example with the sentence `The Prize Pest is a 1951 Warner Bros. Looney Tunes cartoon directed by Robert McKimson, and written by Tedd Pierce.` we only have a single fact `screenwriter(The Prize Pest, Tedd Pierce)`, we added `director(The Prize Pest, Robert McKimson)` which is valid and follows the ontology, but there are hundreds of such cases. This data should be manually reviewed and eventually uploaded to Hugging face datasets.
- addition of `dpedia_webnlg_clean`, where ontologies were modified to remove all camel casing and facts and entities of test/train data were stripped of camelcasing and the use of `_`. The logic behind this is that to train a model on this dataset, it doesn't make sense to ask it to extract facts in camelcase on one part of the dataset (dpedia_webnlg) but not on another (wikidata_tekgen), we should aim have as uniform data as possible. This also avoids technical debt in scripts. 

### File Hierarchy

```
Benchmarker
│   README.md    
│   .env
|
└───data
│   │   
│   │
│   └───dbpedia_webnlg
|   |   |
|   |   └───train
|   |   |       ont_1_train_movie_train.jsonl
|   |   |       ont_2_train_music_train.jsonl
|   |   |       ...
|   |   └───test
|   |   |       ont_1_test_movie_test.jsonl
|   |   |       ont_1_test_music_test.jsonl
|   |   |       ...
|   |   └───ontologies
|   |           ont_1_movie_ontology.json
|   |           ont_2_movie_ontology.json
|   |           ...
|   |
│   └───wikidata_tekgen
|       |
|       └───train
|       |       ont_1_train_university_train.jsonl
|       |       ont_2_train_musicalwork_train.jsonl
|       |       ...
|       └───test
|       |       ont_1_test_university_test.jsonl
|       |       ont_2_test_musicalwork_test.jsonl
|       |       ...
|       |
|       |
|       └───ontologies
|               1_university_ontology.json
|               2_musicalwork_ontology.json
|      
└───evaluation
    │   file021.txt
    │   file022.txt
```


The .env file must contain an `OPEN_AI_KEY` value, if using an `OpenAI` model.
An example of a json object in the `wikidata_tekgen` test sentences files is the following, 

```json
{
    "id": "ont_1_movie_test_1", 
    "sent": "Bleach: Hell Verse (Japanese: BLEACH , Hepburn: Bur\u00c4\u00abchi Jigoku-Hen) is a 2010 Japanese animated film directed by Noriyuki Abe.", 
    "triples": [
        {"sub": "Bleach : Hell Verse", "rel": "director", "obj": "Noriyuki Abe"}, 
        {"sub": "Bleach : Hell Verse", "rel": "publication date", "obj": "01 January 2010"}
    ], 
    "unseen": false, 
    "verified": true, 
    "similars": [
        "ont_1_movie_train_119", 
        "ont_1_movie_train_27", 
        "ont_1_movie_train_67", 
        "ont_1_movie_train_715"
    ]
}
```

with the `dbpedia_webnlg` dataset, the format differs by the lack of the `"unseen"` and `"verified"` keys, as they're not part of this peculiar dataset.
See the original paper's details. Test sentences contain the `"similars"` list of ids of sentences that are similar to the test one in the training set. 

`"unseen": false` means the data comes from some datasource like wikipedia which an LLM may have already seen in pretraining, and `verified` means the 
sentence was manually checked to ensure the triples were extractable. Note that we don't have validation data, indeed the original dataset only 
has some for TekGen, but every sentence only contains one triple, and it's not in the same format as the rest of the data. It seems the authors didn't 
have the time to correctly format it, clean it and generate the triples list. 

## Yggdrasil

View jobs via squeue -u freemana
scontrol show jobid [JOBID]
Care ful not to use = signs in #SBATCH instructions

Run `module load GCCcore/13.2.0 Python/3.11.5 && pipenv shell` to load up python, then `pipenv install && pipenv shell` from the root directory. 
`salloc --ntasks 1 --mem=25G --time=2:00:00 --partition=shared-gpu --gres=gpu:1,VramPerGpu:24G`
YOU HAVE TO SPECIFY MEMORY, OR ELSE IT'LL JUST ALLOCATE 2G, YOU WON'T BE ABLE TO PROCESS THE DATA, check actual memory allocated
with `nvidia-smi` every time, slurm has a tendency not to respect `--mem` parameter, actual VRAM allocated is unknown.
## REBEL Fine tuning

### NYT Fine Tune

To fine tune correctly, update the file paths in `text2kgbench_data.yml` and `text2kgbench.py`.
Update the `relations` array in `score.py`, update the `relations_wikidata_movies` array in 
`lightning_modules.py`. Delete all `.cache` files, update the dataset script filename in
`lightnin_modules.py` at lines 510 and 559, 

Fine tune REBEL on NYT dataset from within the `src` folder with :
```
python3 train.py model=rebel_model data=nyt_data train=nyt_train
```

Try via srun,

```
srun --gpus=1 --mem-per-gpu=32G --partition=shared-gpu --time=1:00:00 python3 train.py model=rebel_model data=text2kgbench_data train=text2kgbench_train
```

```
python3 test.py model=rebel_model data=nyt_data train=nyt_train do_predict=True checkpoint_path='/home/users/f/freemana/Text2KGBenchmarker/experiments/rebel/src/outputs/2024-11-05/12-06-55/experiments/nyt/epoch\=8-step\=21078.ckpt'
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

```
srun --gpus=1 --mem-per-gpu=32G --partition=shared-gpu --time=1:00:00 python3 train.py model=rebel_model data=wikidata_synthetic_data train=wikidata_synthetic_train +trust_remote_code=True
```

