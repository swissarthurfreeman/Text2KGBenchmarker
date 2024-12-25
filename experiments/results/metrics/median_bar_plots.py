import matplotlib.pyplot as plt
import numpy as np

model_name = "Babelscape.rebel-large-"
#model_name = "gpt-4o-"
#name = "shot"
name = "beams"


categories = ["Unseen\nWikidata-TekGen", "Verified\nWikidata-TekGen", "All\nWikidata-TekGen", "All\nDBpedia-WebNLG"]

medians = {}
p75errors = {}
p25errors = {}

metrics = {
    0: "Precision",
    1: "Recall",
    2: "F1"
}
# 0: P, 1: R, 2: F1
metric_idx = 1

# generates bar plots with percentile error margins for F1 score
for i in [2, 4, 6, 8, 10, 12]:
    with open(model_name + f"{i}-{name}-rel-map/global_median.csv") as f:
        text = "".join(f.readlines())
        
        median_table = [line.split(",")[2:] for line in text[text.find("median")+7:text.find("p-25")-5].split("\n")] [1:]
        median_table = [[float(value) for value in line] for line in median_table]
        
        p25_table = [line.split(",")[2:] for line in text[text.find("p-25")+7:text.find("p-75")-5].split("\n")] [1:]
        p25_table = [[float(value) for value in line] for line in p25_table]
        
        p75_table = [line.split(",")[2:] for line in text[text.find("p-75")+7:-2].split("\n")] [1:]
        p75_table = [[float(value) for value in line] for line in p75_table]
        
        # median F1                         Wikidata Unseen,    Wikidata Verified,  Wikidata All,       DBpedia-WebNLG
        medians[f"{i}-{name}"]   = np.round(np.array([ median_table[0][metric_idx], median_table[1][metric_idx], median_table[metric_idx][metric_idx], median_table[3][metric_idx] ]), 2)
        p75errors[f"{i}-{name}"] = np.round(np.array([ p75_table[0][metric_idx], p75_table[1][metric_idx], p75_table[metric_idx][metric_idx], p75_table[3][metric_idx] ]), 2)
        p25errors[f"{i}-{name}"] = np.round(np.array([ p25_table[0][metric_idx], p25_table[1][metric_idx], p25_table[metric_idx][metric_idx], p25_table[3][metric_idx] ]), 2)
   
        


x = np.arange(len(categories))  # the label locations
width = 0.14                     # the width of the bars
multiplier = 0


fig, ax = plt.subplots(layout='constrained')
ax.set_axisbelow(True)
ax.grid(True)
colors = ["#800080", "#55308d", "#2a6099", "#158466", "#00a933", "#81d41a"]

for idx, (nshot, median) in enumerate(medians.items()):
    offset = width * multiplier
    rects = ax.bar(
        x + offset, 
        median, 
        width, 
        label=nshot, 
        yerr=[median-p25errors[nshot], p75errors[nshot]-median], 
        capsize=5,
        color=colors[idx]
    )
    multiplier += 1

print(plt.rcParams["font.sans-serif"])
plt.rcParams["font.family"] = "sans-serif"
plt.rcParams["font.sans-serif"] = ["DejaVu Serif"]

# Add some text for labels, title and custom x-axis tick labels, etc.
ax.set_ylabel(f'Median {metrics[metric_idx]}, 25-th/75-th Percentile')
ax.set_xlabel(f'Text2KGBench Variant')
ax.set_title(f"REBEL with Relation Mapping, {metrics[metric_idx]} Median/percentiles")

ax.set_xticks(x+width*2.5, categories)
ax.legend(loc='upper left', ncols=3)
ax.set_yticks(np.linspace(0, 1, 11))
ax.set_ylim(0, 0.7)

plt.show()

