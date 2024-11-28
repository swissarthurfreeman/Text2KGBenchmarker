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



Output, classic version : 

FORWARD OF LIGHTNING MODULE HERE


inputs torch.Size([26, 116]) torch.Size([26, 116]) torch.Size([26, 145])
labels tensor([[50267,   166,  3561,  ...,  -100,  -100,  -100],
        [50267, 20227,  3320,  ...,     2,  -100,  -100],
        [50267,    83, 13474,  ...,  -100,  -100,  -100],
        ...,
        [50267,  1223,  1115,  ...,  -100,  -100,  -100],
        [50267, 10394,   857,  ...,  -100,  -100,  -100],
        [50267,  3415,  3643,  ...,  -100,  -100,  -100]], device='cuda:0') torch.Size([26, 145])
Output odict_keys(['logits', 'past_key_values', 'decoder_hidden_states', 'encoder_last_hidden_state', 'encoder_hidden_states'])
logits torch.Size([26, 145, 50272]) tensor([[-26.8750,  -1.4141,  -0.6602,  ...,  -3.1719,  -2.6406,  -3.6562],
        [ -5.7812,  -4.6250,   2.5938,  ...,  -4.0000,  -4.0625,  -4.6562],
        [-13.3125,  -3.0625,   6.6562,  ...,  -3.0625,  -3.0625,  -3.0781],
        ...,
        [  2.7812,  -3.2656,  19.5000,  ...,  -2.6562,  -2.8125,  -2.7031],
        [ -7.0312,  -3.9531,   7.0312,  ...,  -4.6562,  -5.3438,  -4.6250],
        [ -9.8125,  -3.9844,   3.5625,  ...,  -3.5312,  -3.8438,  -4.0000]],
       device='cuda:0')


Output, simplified version :

FORWARD OF LIGHTNING MODULE HERE


inputs torch.Size([26, 116]) torch.Size([26, 116]) torch.Size([26, 145])
labels tensor([[    0, 50267,   166,  ...,  -100,  -100,  -100],
        [    0, 50267, 20227,  ...,  8860,     2,  -100],
        [    0, 50267,    83,  ...,  -100,  -100,  -100],
        ...,
        [    0, 50267,  1223,  ...,  -100,  -100,  -100],
        [    0, 50267, 10394,  ...,  -100,  -100,  -100],
        [    0, 50267,  3415,  ...,  -100,  -100,  -100]], device='cuda:0') torch.Size([26, 145])
Output odict_keys(['logits', 'past_key_values', 'encoder_last_hidden_state'])
logits torch.Size([26, 145, 50272]) tensor([[-26.8750,  -1.4141,  -0.6445,  ...,  -3.1719,  -2.6406,  -3.6562],
        [ -5.8438,  -4.6250,   2.7031,  ...,  -3.9688,  -4.0312,  -4.6562],
        [-14.1875,  -4.2812,   5.4062,  ...,  -3.5625,  -3.0156,  -4.0312],
        ...,
        [ -7.4688,  -3.9531,   6.9062,  ...,  -4.6250,  -5.3438,  -4.6250],
        [-10.1250,  -4.0000,   3.3750,  ...,  -3.5156,  -3.8281,  -3.9844],
        [-10.4375,  -0.

Corrected :

FORWARD OF LIGHTNING MODULE HERE


inputs torch.Size([26, 116]) torch.Size([26, 116]) torch.Size([26, 145])
labels tensor([[50267,   166,  3561,  ...,  -100,  -100,  -100],
        [50267, 20227,  3320,  ...,     2,  -100,  -100],
        [50267,    83, 13474,  ...,  -100,  -100,  -100],
        ...,
        [50267,  1223,  1115,  ...,  -100,  -100,  -100],
        [50267, 10394,   857,  ...,  -100,  -100,  -100],
        [50267,  3415,  3643,  ...,  -100,  -100,  -100]], device='cuda:0') torch.Size([26, 145])
Output odict_keys(['logits', 'past_key_values', 'encoder_last_hidden_state'])
logits torch.Size([26, 145, 50272]) tensor([[-26.8750,  -1.4141,  -0.6602,  ...,  -3.1719,  -2.6406,  -3.6562],
        [ -5.7812,  -4.6250,   2.5938,  ...,  -4.0000,  -4.0625,  -4.6562],
        [-13.3125,  -3.0625,   6.6562,  ...,  -3.0625,  -3.0625,  -3.0781],
        ...,
        [  2.7812,  -3.2656,  19.5000,  ...,  -2.6562,  -2.8125,  -2.7031],
        [ -7.0312,  -3.9531,   7.0312,  ...,  -4.6562,  -5.3438,  -4.6250],
        [ -9.8125,  -3.9844,   3.5625,  ...,  -3.5312,  -3.8438,  -4.0000]],
       device='cuda:0')



score
 [array([{'head': 'We Live in Public', 'tail': 'documentary', 'type': 'genre'},
       {'head': 'We Live in Public', 'tail': 'privacy', 'type': 'main subject'},
       {'head': 'We Live in Public', 'tail': 'Grand Jury Prize for Best Documentary', 'type': 'award received'},
       {'head': 'We Live in Public', 'tail': 'Sundance Film Festival', 'type': 'award received'},
       {'head': 'Ondi Timoner', 'tail': 'We Live in Public', 'type': 'notable work'},
       {'head': 'We Live in Public', 'tail': 'Sundance Film Festival Grand Jury Prize for Best Documentary', 'type': 'award received'},
       {'head': 'We Live in Public', 'tail': 'Ondi Timoner', 'type': 'director'}],
      dtype=object), array([{'head': 'Swept Away', 'tail': 'Eros Pagni', 'type': 'cast member'},
       {'head': 'Swept Away', 'tail': 'comedy film', 'type': 'genre'},
       {'head': 'Swept Away', 'tail': 'Isa Danieli', 'type': 'cast member'},
       {'head': 'Swept Away', 'tail': 'Lina Wertmüller', 'type': 'director'},
       {'head': 'Swept Away', 'tail': 'Giancarlo Giannini', 'type': 'cast'},
       {'head': 'Swept Away', 'tail': 'Giancarlo Giannini', 'type': 'cast member'},
       {'head': 'Swept Away', 'tail': 'Mariangela Melato', 'type': 'cast member'},
       {'head': 'Swept Away', 'tail': 'Aldo Puglisi', 'type': 'cast member'},
       {'head': 'Swept Away', 'tail': '1974', 'type': 'publication date'}],
      dtype=object), array([{'head': 'A Gang Story', 'tail': 'Gaumont Film Company', 'type': 'production company'},
       {'head': 'A Gang Story', 'tail': 'Olivier Marchal', 'type': 'director'},
       {'head': 'Olivier Marchal', 'tail': 'France', 'type': 'country of citizenship'},
       {'head': 'A Gang Story', 'tail': 'heist', 'type': 'genre'},
       {'head': 'A Gang Story', 'tail': 'France', 'type': 'country of origin'},
       {'head': 'A Gang Story', 'tail': 'neo-noir', 'type': 'genre'},
       {'head': 'A Gang Story', 'tail': '2011', 'type': 'publication date'}],
      dtype=object), array([{'head': 'Kick-Ass', 'tail': 'Chloë Grace Moretz', 'type': 'cast member'},
       {'head': 'Damon Macready', 'tail': 'Kick-Ass', 'type': 'present in work'},
       {'head': 'Kick-Ass', 'tail': 'Craig Ferguson', 'type': 'cast member'},
       {'head': 'Kick-Ass', 'tail': 'Matthew Vaughn', 'type': 'director'},
       {'head': 'Kick-Ass', 'tail': 'Lyndsy Fonseca', 'type': 'cast member'},
       {'head': 'Kick-Ass', 'tail': 'action-packed film', 'type': 'genre'},
       {'head': 'Kick-Ass', 'tail': 'action', 'type': 'genre'},
       {'head': 'Kick-Ass', 'tail': 'March 12, 2010', 'type': 'publication date'}],
      dtype=object), array([{'head': 'The Intouchables', 'tail': "Olivier Nakache's", 'type': 'director'},
       {'head': 'Olivier Nakache', 'tail': 'The Intouchables', 'type': 'notable work'},
       {'head': 'The Intouchables', 'tail': 'Olivier Nakache', 'type': 'screenwriter'},
       {'head': 'The Intouchables', 'tail': 'European Film Award for Best Film', 'type': 'award received'},
       {'head': 'The Intouchables', 'tail': 'European Film Award for Best Film', 'type': 'nominated for'},
       {'head': 'The Intouchables', 'tail': 'Olivier Nakache', 'type': 'producer'},
       {'head': 'The Intouchables', 'tail': 'Olivier Nakache', 'type': 'cast member'},
       {'head': 'European Film Award for Best Film', 'tail': 'Olivier Nakache', 'type': 'winner'},
       {'head': 'The Intouchables', 'tail': 'Olivier Nakache', 'type': 'director'}],
      dtype=object), array([{'head': 'King Kong Appears in Edo', 'tail': 'fantasy film', 'type': 'genre'},
       {'head': 'King Kong Appears in Edo', 'tail': 'Japan', 'type': 'country of origin'},
       {'head': 'King Kong Appears in Edo', 'tail': 'Edo', 'type': 'narrative location'},
       {'head': 'Sōya Kumagai', 'tail': 'Japan', 'type': 'country of citizenship'},
       {'head': 'King Kong Appears in Edo', 'tail': 'Sōya Kumagai', 'type': 'director'},
       {'head': 'King Kong Appears in Edo', 'tail': 'January 1, 1938', 'type': 'publication date'},
       {'head': 'Sōya Kumagai', 'tail': 'King Kong Appears in Edo', 'type': 'notable work'}],
      dtype=object), array([{'head': 'Bruno Barreto', 'tail': 'Brazil', 'type': 'country of citizenship'},
       {'head': 'Bossa Nova', 'tail': 'Brazil', 'type': 'country of origin'},
       {'head': 'Bossa Nova', 'tail': 'comedy', 'type': 'genre'},
       {'head': 'Bossa Nova', 'tail': 'Bruno Barreto', 'type': 'director'},
       {'head': 'Bossa Nova', 'tail': 'drama', 'type': 'genre'}],
      dtype=object), array([{'head': 'Aitbaar', 'tail': 'Mukul S. Anand', 'type': 'director'},
       {'head': 'Aitbaar', 'tail': 'January 1, 1985', 'type': 'publication date'},
       {'head': 'Aitbaar', 'tail': 'Dimple Kapadia', 'type': 'cast member'},
       {'head': 'Aitbaar', 'tail': 'Vinay Shukla', 'type': 'screenwriter'}],
      dtype=object), array([{'head': 'Hermann Braun', 'tail': 'United States', 'type': 'country of citizenship'},
       {'head': 'Kampfgeschwader Lützow', 'tail': 'war propaganda film', 'type': 'genre'},
       {'head': 'Kampfgeschwader Lützow', 'tail': '1941', 'type': 'publication date'},
       {'head': 'Kampfgeschwader Lützow', 'tail': 'Hermann Braun', 'type': 'cast member'},
       {'head': 'Kampfgeschwader Lützow', 'tail': 'Karl Hermann Martell', 'type': 'cast member'},
       {'head': 'Kampfgeschwader Lützow', 'tail': 'Hans Bertram', 'type': 'director'},
       {'head': 'Kampfgeschwader Lützow', 'tail': 'propaganda film', 'type': 'genre'}],
      dtype=object), array([{'head': 'Put on Ice', 'tail': 'Ángela Molina', 'type': 'cast member'},
       {'head': 'Put on Ice', 'tail': 'Berlin', 'type': 'narrative location'},
       {'head': 'Put on Ice', 'tail': 'Bernhard Sinkel', 'type': 'director'},
       {'head': 'Put on Ice', 'tail': 'Meret Becker', 'type': 'cast member'},
       {'head': 'Put on Ice', 'tail': 'Helmut Griem', 'type': 'cast member'}],
      dtype=object), array([{'head': 'Pancho Villa', 'tail': 'Eugenio Martín', 'type': 'director'},
       {'head': 'Eugenio Martín', 'tail': 'Spain', 'type': 'country of citizenship'},
       {'head': 'Pancho Villa', 'tail': 'Eugenio Martín', 'type': 'screenwriter'},
       {'head': 'Pancho Villa', 'tail': 'Spain', 'type': 'country of origin'},
       {'head': 'Eugenio Martín', 'tail': 'Pancho Villa', 'type': 'notable work'},
       {'head': 'Luis Marín', 'tail': 'Spain', 'type': 'country of citizenship'},
       {'head': 'Pancho Villa', 'tail': 'Luis Marín', 'type': 'cast member'}],
      dtype=object), array([{'head': 'Nest of Vipers', 'tail': 'Capucine', 'type': 'cast member'},
       {'head': 'Nest of Vipers', 'tail': 'Tonino Cervi', 'type': 'director'},
       {'head': 'Nest of Vipers', 'tail': 'Mattia Sbragia', 'type': 'cast member'},
       {'head': 'Nest of Vipers', 'tail': 'drama film', 'type': 'genre'},
       {'head': 'Nest of Vipers', 'tail': 'March 3, 1978', 'type': 'publication date'}],
      dtype=object), array([{'head': '*Places in the Heart*', 'tail': '1984', 'type': 'publication date'},
       {'head': 'Places in the Heart', 'tail': 'Sally Field', 'type': 'cast member'},
       {'head': 'Places in the Heart', 'tail': 'Ed Harris', 'type': 'cast member'},
       {'head': 'Places in the Heart', 'tail': 'John Malkovich', 'type': 'cast member'},
       {'head': 'Robert Benton', 'tail': 'Places in the Heart', 'type': 'notable work'},
       {'head': 'Places in the Heart', 'tail': '1984', 'type': 'publication date'},
       {'head': 'Best Supporting Actor', 'tail': 'Academy Award', 'type': 'instance of'},
       {'head': 'Places in the Heart', 'tail': 'Robert Benton', 'type': 'director'},
       {'head': 'Best Director', 'tail': 'Academy Award', 'type': 'instance of'},
       {'head': '*Places in the Heart*', 'tail': 'Robert Benton', 'type': 'director'}],
      dtype=object), array([{'head': 'Hall Bartlett', 'tail': 'United States', 'type': 'country of citizenship'},
       {'head': 'The Children of Sanchez', 'tail': 'United States', 'type': 'country of origin'},
       {'head': 'The Children of Sanchez', 'tail': 'Hall Bartlett', 'type': 'director'},
       {'head': 'The Children of Sanchez', 'tail': 'Dolores del Rio', 'type': 'cast member'}],
      dtype=object), array([{'head': 'Godzilla', 'tail': 'science fiction film', 'type': 'genre'},
       {'head': 'Godzilla', 'tail': 'Roland Emmerich', 'type': 'director'},
       {'head': 'Godzilla', 'tail': 'September 16, 1998', 'type': 'publication date'},
       {'head': 'Godzilla', 'tail': 'science fiction', 'type': 'genre'},
       {'head': 'Godzilla', 'tail': 'film', 'type': 'instance of'},
       {'head': 'Godzilla', 'tail': 'New York City', 'type': 'narrative location'}],
      dtype=object), array([{'head': 'Henryk Szaro', 'tail': 'The Year 1914', 'type': 'notable work'},
       {'head': 'The Year 1914', 'tail': 'Wacław Sieroszewski', 'type': 'screenwriter'},
       {'head': 'The Year 1914', 'tail': 'Jadwiga Smosarska', 'type': 'cast member'},
       {'head': 'The Year 1914', 'tail': 'Jan Kurnakowicz', 'type': 'cast member'},
       {'head': 'The Year 1914', 'tail': '1932', 'type': 'publication date'},
       {'head': 'The Year 1914', 'tail': 'Henryk Szaro', 'type': 'director'},
       {'head': 'The Year 1914', 'tail': 'Witold Conti', 'type': 'cast member'}],
      dtype=object)] 
 [[{'head': 'We Live in Public', 'type': 'director', 'tail': 'Ondi Timoner'}, {'head': 'We Live in Public', 'type': 'main subject', 'tail': 'privacy'}, {'head': 'We Live in Public', 'type': 'award received', 'tail': 'Sundance Film Festival Grand Jury Prize for Best Documentary'}], [{'head': 'Swept Away', 'type': 'director', 'tail': 'Lina Wertmüller'}, {'head': 'Swept Away', 'type': 'cast member', 'tail': 'Mariangela Melato'}, {'head': 'Swept Away', 'type': 'country of origin', 'tail': 'Italy'}, {'head': 'Swept Away', 'type': 'publication date', 'tail': '1974-01-01T00:00:00Z'}, {'head': 'Swept Away', 'type': 'cast member', 'tail': 'Giancarlo Giannini'}, {'head': 'Swept Away', 'type': 'genre', 'tail': 'comedy film'}, {'head': 'Swept Away', 'type': 'production company', 'tail': 'Medusa Film'}, {'head': 'Swept Away', 'type': 'cast member', 'tail': 'Eros Pagni'}, {'head': 'Swept Away', 'type': 'screenwriter', 'tail': 'Lina Wertmüller'}, {'head': 'Swept Away', 'type': 'cast member', 'tail': 'Aldo Puglisi'}, {'head': 'Swept Away', 'type': 'cast member', 'tail': 'Isa Danieli'}, {'head': 'Lina Wertmüller', 'type': 'country of citizenship', 'tail': 'Italy'}], [{'head': 'A Gang Story', 'type': 'director', 'tail': 'Olivier Marchal'}, {'head': 'A Gang Story', 'type': 'production company', 'tail': 'Gaumont Film Company'}, {'head': 'A Gang Story', 'type': 'genre', 'tail': 'heist film'}, {'head': 'A Gang Story', 'type': 'genre', 'tail': 'neo-noir'}, {'head': 'A Gang Story', 'type': 'country of origin', 'tail': 'France'}, {'head': 'A Gang Story', 'type': 'publication date', 'tail': '2011-11-10T00:00:00Z'}, {'head': 'Olivier Marchal', 'type': 'country of citizenship', 'tail': 'France'}], [{'head': 'Kick-Ass', 'type': 'director', 'tail': 'Matthew Vaughn'}, {'head': 'Kick-Ass', 'type': 'genre', 'tail': 'action film'}, {'head': 'Kick-Ass', 'type': 'characters', 'tail': 'Damon Macready'}, {'head': 'Kick-Ass', 'type': 'cast member', 'tail': 'Chloë Grace Moretz'}, {'head': 'Kick-Ass', 'type': 'production company', 'tail': 'Marv Studios'}, {'head': 'Kick-Ass', 'type': 'cast member', 'tail': 'Lyndsy Fonseca'}, {'head': 'Kick-Ass', 'type': 'publication date', 'tail': '2010-03-12T00:00:00Z'}, {'head': 'Kick-Ass', 'type': 'cast member', 'tail': 'Craig Ferguson'}], [{'head': 'The Intouchables', 'type': 'director', 'tail': 'Olivier Nakache'}, {'head': 'The Intouchables', 'type': 'genre', 'tail': 'drama film'}, {'head': 'The Intouchables', 'type': 'nominated for', 'tail': 'European Film Award for Best Film'}, {'head': 'The Intouchables', 'type': 'genre', 'tail': 'film based on literature'}, {'head': 'The Intouchables', 'type': 'nominated for', 'tail': 'International Submission to the Academy Awards'}], [{'head': 'King Kong Appears in Edo', 'type': 'director', 'tail': 'Sōya Kumagai'}, {'head': 'King Kong Appears in Edo', 'type': 'genre', 'tail': 'fantasy film'}, {'head': 'King Kong Appears in Edo', 'type': 'country of origin', 'tail': 'Japan'}, {'head': 'King Kong Appears in Edo', 'type': 'publication date', 'tail': '1938-01-01T00:00:00Z'}, {'head': 'King Kong Appears in Edo', 'type': 'narrative location', 'tail': 'Edo'}], [{'head': 'Bruno Barreto', 'type': 'country of citizenship', 'tail': 'Brazil'}, {'head': 'Bossa Nova', 'type': 'director', 'tail': 'Bruno Barreto'}, {'head': 'Bossa Nova', 'type': 'genre', 'tail': 'comedy drama'}, {'head': 'Bossa Nova', 'type': 'genre', 'tail': 'drama film'}], [{'head': 'Aitbaar', 'type': 'director', 'tail': 'Mukul S. Anand'}, {'head': 'Aitbaar', 'type': 'screenwriter', 'tail': 'Vinay Shukla'}, {'head': 'Aitbaar', 'type': 'cast member', 'tail': 'Dimple Kapadia'}, {'head': 'Aitbaar', 'type': 'publication date', 'tail': '1985-01-01T00:00:00Z'}], [{'head': 'Kampfgeschwader Lützow', 'type': 'director', 'tail': 'Hans Bertram'}, {'head': 'Kampfgeschwader Lützow', 'type': 'screenwriter', 'tail': 'Wolf Neumeister'}, {'head': 'Kampfgeschwader Lützow', 'type': 'cast member', 'tail': 'Karl Hermann Martell'}, {'head': 'Kampfgeschwader Lützow', 'type': 'genre', 'tail': 'propaganda film'}, {'head': 'Kampfgeschwader Lützow', 'type': 'genre', 'tail': 'war film'}, {'head': 'Kampfgeschwader Lützow', 'type': 'cast member', 'tail': 'Hermann Braun'}, {'head': 'Kampfgeschwader Lützow', 'type': 'publication date', 'tail': '1941-01-01T00:00:00Z'}, {'head': 'Karl Hermann Martell', 'type': 'country of citizenship', 'tail': 'Germany'}, {'head': 'Hermann Braun', 'type': 'country of citizenship', 'tail': 'United States of America'}], [{'head': 'Put on Ice', 'type': 'director', 'tail': 'Bernhard Sinkel'}, {'head': 'Put on Ice', 'type': 'cast member', 'tail': 'Ángela Molina'}, {'head': 'Put on Ice', 'type': 'cast member', 'tail': 'Helmut Griem'}, {'head': 'Put on Ice', 'type': 'cast member', 'tail': 'Meret Becker'}, {'head': 'Put on Ice', 'type': 'narrative location', 'tail': 'Berlin'}], [{'head': 'Pancho Villa', 'type': 'director', 'tail': 'Eugenio Martín'}, {'head': 'Pancho Villa', 'type': 'country of origin', 'tail': 'Spain'}, {'head': 'Pancho Villa', 'type': 'cast member', 'tail': 'Luis Marín'}, {'head': 'Pancho Villa', 'type': 'screenwriter', 'tail': 'Eugenio Martín'}], [{'head': 'Nest of Vipers', 'type': 'director', 'tail': 'Tonino Cervi'}, {'head': 'Nest of Vipers', 'type': 'cast member', 'tail': 'Mattia Sbragia'}, {'head': 'Nest of Vipers', 'type': 'publication date', 'tail': '1979-03-01T00:00:00Z'}, {'head': 'Nest of Vipers', 'type': 'publication date', 'tail': '1978-11-17T00:00:00Z'}, {'head': 'Nest of Vipers', 'type': 'cast member', 'tail': 'Capucine'}, {'head': 'Nest of Vipers', 'type': 'genre', 'tail': 'drama film'}, {'head': 'Nest of Vipers', 'type': 'publication date', 'tail': '1978-10-23T00:00:00Z'}, {'head': 'Nest of Vipers', 'type': 'publication date', 'tail': '1978-07-09T00:00:00Z'}, {'head': 'Nest of Vipers', 'type': 'publication date', 'tail': '1978-03-03T00:00:00Z'}], [{'head': 'Places in the Heart', 'type': 'director', 'tail': 'Robert Benton'}, {'head': 'Places in the Heart', 'type': 'nominated for', 'tail': 'Academy Award for Best Supporting Actor'}, {'head': 'Places in the Heart', 'type': 'cast member', 'tail': 'John Malkovich'}, {'head': 'Places in the Heart', 'type': 'genre', 'tail': 'drama film'}, {'head': 'Places in the Heart', 'type': 'cast member', 'tail': 'Ed Harris'}, {'head': 'Places in the Heart', 'type': 'publication date', 'tail': '1984-01-01T00:00:00Z'}, {'head': 'Places in the Heart', 'type': 'nominated for', 'tail': 'Academy Award for Best Director'}, {'head': 'Places in the Heart', 'type': 'award received', 'tail': 'National Board of Review: Top Ten Films'}, {'head': 'Places in the Heart', 'type': 'cast member', 'tail': 'Sally Field'}, {'head': 'Places in the Heart', 'type': 'publication date', 'tail': '1985-03-01T00:00:00Z'}, {'head': 'Places in the Heart', 'type': 'screenwriter', 'tail': 'Robert Benton'}], [{'head': 'The Children of Sanchez', 'type': 'director', 'tail': 'Hall Bartlett'}, {'head': 'The Children of Sanchez', 'type': 'country of origin', 'tail': 'United States of America'}, {'head': 'The Children of Sanchez', 'type': 'cast member', 'tail': 'Dolores del Rio'}], [{'head': 'Godzilla', 'type': 'director', 'tail': 'Roland Emmerich'}, {'head': 'Godzilla', 'type': 'publication date', 'tail': '1998-09-16T00:00:00Z'}, {'head': 'Godzilla', 'type': 'main subject', 'tail': 'dinosaur'}, {'head': 'Godzilla', 'type': 'genre', 'tail': 'science fiction film'}, {'head': 'Godzilla', 'type': 'screenwriter', 'tail': 'Roland Emmerich'}, {'head': 'Godzilla', 'type': 'narrative location', 'tail': 'New York City'}], [{'head': 'The Year 1914', 'type': 'director', 'tail': 'Henryk Szaro'}, {'head': 'The Year 1914', 'type': 'cast member', 'tail': 'Jan Kurnakowicz'}, {'head': 'The Year 1914', 'type': 'cast member', 'tail': 'Jadwiga Smosarska'}, {'head': 'The Year 1914', 'type': 'cast member', 'tail': 'Witold Conti'}, {'head': 'The Year 1914', 'type': 'country of origin', 'tail': 'Poland'}, {'head': 'The Year 1914', 'type': 'screenwriter', 'tail': 'Wacław Sieroszewski'}, {'head': 'The Year 1914', 'type': 'publication date', 'tail': '1932-01-01T00:00:00Z'}, {'head': 'The Year 1914', 'type': 'genre', 'tail': 'war film'}, {'head': 'Henryk Szaro', 'type': 'country of citizenship', 'tail': 'Poland'}, {'head': 'Jan Kurnakowicz', 'type': 'country of citizenship', 'tail': 'Poland'}, {'head': 'Jadwiga Smosarska', 'type': 'country of citizenship', 'tail': 'Poland'}]]
relations ['director', 'screenwriter', 'genre', 'based on', 'cast member', 'award received', 'production company', 'country of origin', 'publication date', 'characters', 'narrative location', 'filming location', 'main subject', 'nominated for', 'cost']



Error executing job with overrides: ['model=rebel_model', 'data=wkdata_synth_movie']
Traceback (most recent call last):
  File "/home/users/f/freemana/Text2KGBenchmarker/experiments/bench-rebel/src/train.py", line 86, in main
    train(conf)
  File "/home/users/f/freemana/Text2KGBenchmarker/experiments/bench-rebel/src/train.py", line 79, in train
    trainer.fit(pl_module, datamodule=pl_data_module)
  File "/home/users/f/freemana/.local/share/virtualenvs/Text2KGBenchmarker-yg4X5boN/lib/python3.11/site-packages/pytorch_lightning/trainer/trainer.py", line 538, in fit
    call._call_and_handle_interrupt(
  File "/home/users/f/freemana/.local/share/virtualenvs/Text2KGBenchmarker-yg4X5boN/lib/python3.11/site-packages/pytorch_lightning/trainer/call.py", line 47, in _call_and_handle_interrupt
    return trainer_fn(*args, **kwargs)
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/home/users/f/freemana/.local/share/virtualenvs/Text2KGBenchmarker-yg4X5boN/lib/python3.11/site-packages/pytorch_lightning/trainer/trainer.py", line 574, in _fit_impl
    self._run(model, ckpt_path=ckpt_path)
  File "/home/users/f/freemana/.local/share/virtualenvs/Text2KGBenchmarker-yg4X5boN/lib/python3.11/site-packages/pytorch_lightning/trainer/trainer.py", line 981, in _run
    results = self._run_stage()
              ^^^^^^^^^^^^^^^^^
  File "/home/users/f/freemana/.local/share/virtualenvs/Text2KGBenchmarker-yg4X5boN/lib/python3.11/site-packages/pytorch_lightning/trainer/trainer.py", line 1023, in _run_stage
    self._run_sanity_check()
  File "/home/users/f/freemana/.local/share/virtualenvs/Text2KGBenchmarker-yg4X5boN/lib/python3.11/site-packages/pytorch_lightning/trainer/trainer.py", line 1052, in _run_sanity_check
    val_loop.run()
  File "/home/users/f/freemana/.local/share/virtualenvs/Text2KGBenchmarker-yg4X5boN/lib/python3.11/site-packages/pytorch_lightning/loops/utilities.py", line 178, in _decorator
    return loop_run(self, *args, **kwargs)
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/home/users/f/freemana/.local/share/virtualenvs/Text2KGBenchmarker-yg4X5boN/lib/python3.11/site-packages/pytorch_lightning/loops/evaluation_loop.py", line 142, in run
    return self.on_run_end()
           ^^^^^^^^^^^^^^^^^
  File "/home/users/f/freemana/.local/share/virtualenvs/Text2KGBenchmarker-yg4X5boN/lib/python3.11/site-packages/pytorch_lightning/loops/evaluation_loop.py", line 268, in on_run_end
    self._on_evaluation_end()
  File "/home/users/f/freemana/.local/share/virtualenvs/Text2KGBenchmarker-yg4X5boN/lib/python3.11/site-packages/pytorch_lightning/loops/evaluation_loop.py", line 314, in _on_evaluation_end
    call._call_lightning_module_hook(trainer, hook_name, *args, **kwargs)
  File "/home/users/f/freemana/.local/share/virtualenvs/Text2KGBenchmarker-yg4X5boN/lib/python3.11/site-packages/pytorch_lightning/trainer/call.py", line 167, in _call_lightning_module_hook
    output = fn(*args, **kwargs)
             ^^^^^^^^^^^^^^^^^^^
  File "/home/users/f/freemana/Text2KGBenchmarker/experiments/bench-rebel/src/lightning_module.py", line 331, in on_validation_end
    precision, recall, f1 = re_score(
                            ^^^^^^^^^
  File "/home/users/f/freemana/Text2KGBenchmarker/experiments/bench-rebel/src/metrics.py", line 41, in re_score
    pred_rels = {(rel["head"], rel["tail"]) for rel in pred_sent if rel["type"] == rel_type}
                ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/home/users/f/freemana/Text2KGBenchmarker/experiments/bench-rebel/src/metrics.py", line 41, in <setcomp>
    pred_rels = {(rel["head"], rel["tail"]) for rel in pred_sent if rel["type"] == rel_type}
                                                                    ~~~^^^^^^^^
IndexError: only integers, slices (`:`), ellipsis (`...`), numpy.newaxis (`None`) and integer or boolean arrays are valid indices