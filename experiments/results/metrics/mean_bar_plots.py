import matplotlib.pyplot as plt
import numpy as np
import json
from pprint import pprint
model_name = "gpt-3.5-turbo-"
#name = "shot"
#model_name = "gpt-4o-"
model_name = "Babelscape.rebel-large-"
name = "beams"

categories = ["Unseen\nWikidata-TekGen", "Verified\nWikidata-TekGen", "All\nWikidata-TekGen", "All\nDBpedia-WebNLG"]
metric = "avg_recall"

means = {"Vicuna": [0.32, 0.38, 0.35, 0.30]}
means = {}
stds = {"Vicuna": [0, 0, 0, 0]}
stds = {}

# generates bar plots with percentile error margins for F1 score
for i in [2, 4, 6, 8, 10, 12]:
    
    # average global F1 for this model, position zero is unseen, 1 is verified, 2 is all, 3 is dbpedia all
    mean_F1 = []
    std_F1 = []
    
    with open(model_name + f"{i}-{name}-rel-map/wikidata_tekgen_avg.jsonl") as f:
        data = [json.loads(line) for line in f]
        
        F1s_unseen, F1s_verified, F1s_all = [], [], []
        for onto in data:
            F1s_unseen.append(onto["unseen"][metric])
            F1s_verified.append(onto["verified"][metric])
            F1s_all.append(onto["all"][metric])
        
        mean_F1.append(np.mean(F1s_unseen))
        std_F1.append(np.std(F1s_unseen))
        
        mean_F1.append(np.mean(F1s_verified))
        std_F1.append(np.std(F1s_verified))
        
        mean_F1.append(np.mean(F1s_all))
        std_F1.append(np.std(F1s_all))
        
    with open(model_name + f"{i}-{name}-rel-map/dpedia_webnlg_clean_avg.jsonl") as f:
        data = [json.loads(line) for line in f]
        
        F1s = []
        for onto in data:
            F1s.append(onto["all"][metric])
        
        mean_F1.append(np.mean(F1s))
        std_F1.append(np.std(F1s))
    
    means[f"{i}-{name}"]   = np.round(np.array(mean_F1), 2)
    stds[f"{i}-{name}"] = np.round(np.array(std_F1), 2)
        
        

pprint(means)
pprint(stds)

x = np.arange(len(categories))  # the label locations
width = 0.13                     # the width of the bars
multiplier = 0


fig, ax = plt.subplots(layout='constrained')
ax.set_axisbelow(True)
ax.grid(True)
colors = ["#800080", "#55308d", "#2a6099", "#158466", "#00a933", "#81d41a"]

for idx, (nshot, mean) in enumerate(means.items()):
    offset = width * multiplier
    rects = ax.bar(
        x + offset, 
        mean, 
        width, 
        label=nshot, 
        yerr=[stds[nshot]], 
        capsize=5,
        color=colors[idx]
    )
    multiplier += 1

plt.rcParams["font.family"] = "sans-serif"
plt.rcParams["font.sans-serif"] = ["DejaVu Serif"]

ax.set_title(f"REBEL-large, F1 Recall and Standard Deviation across Ontologies")
ax.set_ylabel('Mean Recall, standard deviation bars.')
ax.set_xlabel('Text2KGBench Variant')

ax.set_xticks(x+width*2.5, categories)
ax.legend(loc='upper left', ncols=3)
ax.set_yticks(np.linspace(0, 1, 11))
ax.set_ylim(0, 0.7)

plt.show()

