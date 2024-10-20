## Benchmarker : Text2KGBench

This is a Text2KGBench cleaned dataset with a rerwite of utility functions to fix technical debt
from the original repository to make working with the dataset difficult easier. 

The main change is simply that the data is all kept in `jsonl` files in the following simple
hierarchy instead of scattered across multiple jsonl folders, `jsonl` and `txt` files. This
data should probably be eventually uploaded to Hugging face datasets.

Another sizeable change is the modification of the ontology files to remove ghost wikidata IDs
which did not have the corresponding property or entity name within the ontology file and the
addition of missing domain and range classes to certain entities, to avoid for instance, describing
a publication date via `publication_date(film,)`, instead we add `date` as range to yield
`publication_date(film, date)`. 

### File Hierarchy

```
Benchmarker
│   README.md    
│
└───data
│   │   
│   │
│   └───dbpedia_webnlg
|   |   |
|   |   └───train
|   |   |       ont_1_train_movie_sentences.jsonl
|   |   |       ont_2_train_music_sentences.jsonl
|   |   |       ...
|   |   └───test
|   |   |       ont_1_test_movie_sentences.jsonl
|   |   |       ont_1_test_music_sentences.jsonl
|   |   |       ...
|   |   └───ontologies
|   |           ont_1_movie_ontology.json
|   |           ont_2_movie_ontology.json
|   |           ...
|   |
│   └───wikidata_tekgen
|       |
|       └───train
|       |       ont_1_train_university_sentences.jsonl
|       |       ont_2_train_musicalwork_sentences.jsonl
|       |       ...
|       └───test
|       |       ont_1_test_university_sentences.jsonl
|       |       ont_2_test_musicalwork_sentences.jsonl
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
        "ont_1_movie_train_27", 
        "ont_1_movie_train_612", 
        "ont_1_movie_train_715", 
        "ont_1_movie_train_67", 
        "ont_1_movie_train_119"
    ]
}
```

with the `dbpedia_webnlg` dataset, the format differs by the lack of the `"unseen"` and `"verified"` keys, as they're not part of this peculiar dataset.
See the original paper's details. Test sentences contain the `"similars"` list of ids of sentences that are similar to the test one in the training set. 

`"unseen": true` means the data comes from some datasource like wikipedia which an LLM may have already seen in pretraining, and `verified` means the 
sentence was manually checked to ensure the triples were extractable. Note that we don't have validation data, indeed the original dataset only 
has some for TekGen, but every sentence only contains one triple, and it's not in the same format as the rest of the data. It seems the authors didn't 
have the time to correctly format it, clean it and generate the triples list. 

##