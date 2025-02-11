import json
import matplotlib.pyplot as plt

F1_dbpedia = []
F1_wikidata = []

for i in [1, 2, 3, 4, 5, 6]:
    for model_name in [f'gpt-4o-{i}-shot', f'gpt-3.5-turbo-{i}-shot']:
            
        with open("../results/metrics/" + model_name + "/dbpedia_webnlg_clean_avg.jsonl") as f:
            data_dbpedia = [json.loads(line) for line in f]
            for onto_res in data_dbpedia:
                F1_dbpedia.append(onto_res["all"]["avg_f1"])
        
        with open("../results/metrics/" + model_name + "/wikidata_tekgen_avg.jsonl") as f:
            data_wikidata = [json.loads(line) for line in f]
            for onto_res in data_wikidata:
                F1_wikidata.append(onto_res["all"]["avg_f1"])
            
            
#with open("./F1_list.txt", "w") as f:
#    f.write("dbpedia_F1 :" + str(F1_dbpedia) + "\n")
#    f.write("wikidata_F1 :" + str(F1_wikidata))
    

dbpedia_f1s_gpt35and4oAllShots = F1_dbpedia
wikidata_all_f1s_gpt35and4oAllShots = F1_wikidata

plt.rcParams["font.family"] = "sans-serif"
plt.rcParams["font.sans-serif"] = ["DejaVu Serif"]

ax = plt.subplot(1, 2, 1)
ax.set_axisbelow(True)
ax.grid(True)
plt.title("Wikidata-TekGen F1 Occurences, all shots, GPT-3.5/4o")
plt.xlabel("F1 Score\nWikidata-TekGen All")
plt.hist(wikidata_all_f1s_gpt35and4oAllShots, bins=10)

ax = plt.subplot(1, 2, 2)
ax.set_axisbelow(True)
ax.grid(True)
plt.title("DBpedia-WebNLG F1 Occurences, all shots, GPT-3.5/4o")
plt.xlabel("F1 Score\nDBpedia-WebNLG")
plt.hist(dbpedia_f1s_gpt35and4oAllShots, bins=10)
plt.show()
