import json


F1_dbpedia = []
F1_wikidata = []

for i in [1, 2, 3, 4, 5, 6]:
    for model_name in [f'gpt-4o-{i}-shot', f'gpt-3.5-turbo-{i}-shot']:
            
        with open(model_name + "/dpedia_webnlg_clean_avg.jsonl") as f:
            data_dbpedia = [json.loads(line) for line in f]
            for onto_res in data_dbpedia:
                F1_dbpedia.append(onto_res["all"]["avg_f1"])
        
        with open(model_name + "/wikidata_tekgen_avg.jsonl") as f:
            data_wikidata = [json.loads(line) for line in f]
            for onto_res in data_wikidata:
                F1_wikidata.append(onto_res["all"]["avg_f1"])
            
            
with open("./F1_list.txt", "w") as f:
    f.write("dbpedia_F1 :" + str(F1_dbpedia) + "\n")
    f.write("wikidata_F1 :" + str(F1_wikidata))