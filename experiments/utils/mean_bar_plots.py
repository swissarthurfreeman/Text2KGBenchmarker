import os
import json
import glob
import argparse
import numpy as np
from pprint import pprint
import matplotlib.pyplot as plt


def get_means_stds(llm_metric_folder_path: str, metric: str) -> tuple[list[float], list[float]]:
    """Compute mean and standard deviations for metrics inside folder. Return means and stds 
    list of 4 (with dbpedia) or 3 values corresponding to means or stds."""
    means = {}
    stds = {}
    
    mean_F1 = []
    std_F1 = []
    
    with open(llm_metric_folder_path + "/wikidata_tekgen_avg.jsonl") as f:
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
    
    if os.path.exists(llm_metric_folder_path + "/dpedia_webnlg_clean_avg.jsonl"):
        with open(llm_metric_folder_path + "/dpedia_webnlg_clean_avg.jsonl") as f:
            data = [json.loads(line) for line in f]
            
            F1s = []
            for onto in data:
                F1s.append(onto["all"][metric])
            
            mean_F1.append(np.mean(F1s))
            std_F1.append(np.std(F1s))
    else:
        means['Vicuna'].pop()
        stds['Vicuna'].pop()
    
    return np.round(np.array(mean_F1), 2), np.round(np.array(std_F1), 2)
    

def gen_bar_plots(metric_paths: list[str], plot_labels: list[str], title_model_name: str, metric: str, categories: list[str], ylim: float = 0.7) -> None:
    """Generate a graph with a bar for every model in `metrics_paths` list,
    bar will plot specified `metric` mean across ontologies and standard deviation."""
    assert len(metric_paths) == len(plot_labels), "Every bar plot must have a label !"
    
    colors = ["#800080"]
    start = 5582989
    
    for _ in range(len(metric_paths)+1):
        colors.append("#" + hex(start)[2:])
        start += 43300

    fig, ax = plt.subplots(layout='constrained')
    ax.set_axisbelow(True)
    ax.grid(True)
    
    means_vicuna = [0.32, 0.38, 0.35, 0.30]
    stds_vicuna  = [0, 0, 0, 0]
    
    if len(categories) < 4: # if no DBpedia-WebNLG category (like with finetuned REBEL)
        means_vicuna.pop()
        stds_vicuna.pop()
    
    x = np.arange(len(categories))  # the label locations
    width = 0.13                     # the width of the bars
    
    # plot Vicuna baseline bar
    ax.bar(x, means_vicuna, width, label='Vicuna', yerr=[stds_vicuna], capsize=5, color=colors[0])
    multiplier = 1
    
    # llm_metric_folder_path is gpt-4o-4-shot for example
    # for every specified model metric folder, plot one bar for every variant
    for idx, (llm_metric_folder_path, plot_label) in enumerate(zip(metric_paths, plot_labels)):
        
        offset = width * multiplier
        
        means, stds = get_means_stds(llm_metric_folder_path, metric)
        
        print("bar", plot_label, means, stds, multiplier)
        ax.bar(x + offset, means, width, label=plot_label, yerr=[stds], capsize=5, color=colors[idx+1])
        multiplier += 1
    
    plt.rcParams["font.family"] = "sans-serif"
    plt.rcParams["font.sans-serif"] = ["DejaVu Serif"]

    ax.set_title(f"{title_model_name}, Mean {metric} & Standard Deviation")
    ax.set_ylabel(f'Mean {metric.upper()}, standard deviation bars.')
    ax.set_xlabel('Text2KGBench Variant')

    ax.set_xticks(x+width*2.5, categories)
    ax.legend(loc='upper left', ncols=3)
    ax.set_yticks(np.linspace(0, 1, 11))
    ax.set_ylim(0, ylim)
                
    plt.savefig('./test.png')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Generate mean bar plots for specified models and metric with standard deviation errors.')
    parser.add_argument("--title_model_name", type=str, default='Default Title')
    parser.add_argument("--model_name_glob", type=str, default='')
    parser.add_argument("--model_names_list", type=str, nargs="*", default=[])
    parser.add_argument("--plot_labels", type=str, nargs="*", default=[])
    
    parser.add_argument("--variants", type=str, nargs="*", default=["Unseen\nWikidata-TekGen", "Verified\nWikidata-TekGen", "All\nWikidata-TekGen", "All\nDBpedia-WebNLG"])
    parser.add_argument("--metric", type=str, default="avg_f1", choices=['avg_f1', 'avg_recall', 'avg_precision'])
    
    args = parser.parse_args()

    print(args.title_model_name)
    print(args.model_names_list)
    print(args.model_name_glob)
    print(args.plot_labels)
    
    if len(args.model_names_list) == 0 and args.model_name_glob == '':
        print("No models list or model name glob expression specified, aborting.")
        exit(0)
    
    llm_metric_folders = []
    if args.model_name_glob != '':
        llm_metric_folders = glob.glob("../results/metrics/" + args.model_name_glob)
    else:
        for model_name in args.model_names_list:
            llm_metric_folders.append("../results/metrics/" + model_name)
    
    llm_metric_folders = list(sorted(llm_metric_folders))
    
    plot_labels = []
    if len(args.plot_labels) == 0:
        print("plot_labels not provided, inferring from folder names...")
        a_folder_name = llm_metric_folders[0].split("/")[-1] 
        if 'beams' in a_folder_name:
            for path in llm_metric_folders:
                splitted_folder_name: list[str] = path.split("/")[-1].split("-")
                idx = splitted_folder_name.index("beams")
                plot_labels.append(f"{splitted_folder_name[idx-1]}-beams")
            
        elif 'shot' in a_folder_name:
            for path in llm_metric_folders:
                splitted_folder_name: list[str] = path.split("/")[-1].split("-")
                idx = splitted_folder_name.index("shot")
                plot_labels.append(f"{splitted_folder_name[idx-1]}-shot")
        
        elif 't=' in a_folder_name:
            for path in llm_metric_folders:
                tequals: str = path.split("/")[-1].split("-")[-1]
                plot_labels.append(f"{tequals[2:]}")
        
        else:
            print("Unable to infer from folder names, aborting...")
            exit(0)
    else:
        plot_labels = args.plot_labels
        
    
    print("Generating bar plots for metrics inside : ")
    pprint(llm_metric_folders)
    
    print("Using plot labels :", plot_labels)
    
    gen_bar_plots(llm_metric_folders, plot_labels, args.title_model_name, args.metric, args.variants, ylim=1)
    
