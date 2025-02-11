
The list of changes is as follows : 

- data is all kept in `jsonl` files in the hierarchy below, instead of a mix of data across multiple `jsonl` and `txt` files.
- ontology files were edited to remove ghost wikidata IDs which did not a corresponding property or entity label within the file.
- addition of missing domain and range classes to certain properties, to avoid for instance, describing a publication date via `publication_date(film,)`, instead we add `date` as range to yield `publication_date(film, date)`.
- folded all duplicate sentences in `wikidata_tekgen` test and train datasets, aggregating into the same list the facts from train. Indeed, in the train files, certain objects with different ids had the same sentences but with different facts, they were folded into a single id and their facts were concatenated into a list following the same format as `dbpedia_webnlg`. The same processing was done on the `test` data of `wikidata_tekgen` where additionally the `similars` list was updated with the new id folds, replacing the removed sentences by the id of the sentence folded to.  
- `wikidata_tekgen` dataset still has many problems. There are a lot of triples missing, for example with the sentence `The Prize Pest is a 1951 Warner Bros. Looney Tunes cartoon directed by Robert McKimson, and written by Tedd Pierce.` we only have a single fact `screenwriter(The Prize Pest, Tedd Pierce)`, we added `director(The Prize Pest, Robert McKimson)` which is valid and follows the ontology, but there are hundreds of such cases. This data should be manually reviewed and eventually uploaded to Hugging face datasets.
- addition of `dbpedia_webnlg_clean`, where ontologies were modified to remove all camel casing and facts and entities of test/train data were stripped of camelcasing and the use of `_`. The logic behind this is that to train a model on this dataset, it doesn't make sense to ask it to extract facts in camelcase on one part of the dataset (dbpedia_webnlg) but not on another (wikidata_tekgen), we should aim have as uniform data as possible. This also avoids technical debt in scripts. 

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

