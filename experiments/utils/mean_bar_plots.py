import os
import json
import glob
import argparse
import numpy as np
from pprint import pprint
import matplotlib.pyplot as plt
from distutils.util import strtobool

def get_median_percentiles(llm_metric_folder_path: str, metric: str):
    metric_idx: int = {
        "avg_precision": 0,
        "avg_recall": 1,
        "avg_f1": 2
    }[metric]

    if not os.path.exists(llm_metric_folder_path + "/global_median.csv"):
        print("global_median.csv file doesn't exist at", llm_metric_folder_path + "/global_median.csv, aborting...")
        exit(1)
        
    with open(llm_metric_folder_path + "/global_median.csv") as f:
        text = "".join(f.readlines())
        
        median_table = [line.split(",")[2:] for line in text[text.find("median")+7:text.find("p-25")-5].split("\n")] [1:]
        median_table = [[float(value) for value in line] for line in median_table]
        
        p25_table = [line.split(",")[2:] for line in text[text.find("p-25")+7:text.find("p-75")-5].split("\n")] [1:]
        p25_table = [[float(value) for value in line] for line in p25_table]
        
        p75_table = [line.split(",")[2:] for line in text[text.find("p-75")+7:-2].split("\n")] [1:]
        p75_table = [[float(value) for value in line] for line in p75_table]
        
        # median F1                         Wikidata Unseen,    Wikidata Verified,  Wikidata All,       DBpedia-WebNLG
        medians   = np.round(np.array([ median_table[0][metric_idx], median_table[1][metric_idx], median_table[metric_idx][metric_idx], median_table[3][metric_idx] ]), 2)
        p75errors = np.round(np.array([ p75_table[0][metric_idx], p75_table[1][metric_idx], p75_table[metric_idx][metric_idx], p75_table[3][metric_idx] ]), 2)
        p25errors = np.round(np.array([ p25_table[0][metric_idx], p25_table[1][metric_idx], p25_table[metric_idx][metric_idx], p25_table[3][metric_idx] ]), 2)

        return medians, p75errors, p25errors


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
    

def gen_bar_plots(metric_paths: list[str], plot_labels: list[str], title_model_name: str, metric: str, categories: list[str], ylim: float = 0.7, vicuna_f1_bars: bool = True, mode: str = 'mean') -> None:
    """Generate a graph with a bar for every model in `metrics_paths` list,
    bar will plot specified `metric` mean across ontologies and standard deviation."""
    assert len(metric_paths) == len(plot_labels), "Every bar plot must have a label !"
    
    #start = 5582989
    
    #for _ in range(len(metric_paths)+1):
    #    colors.append("#" + hex(start)[2:])
    #    start += 43300

    fig, ax = plt.subplots(layout='constrained')
    ax.set_axisbelow(True)
    ax.grid(True)
    x = np.arange(len(categories))   # the label locations
    width = 0.13                     # the width of the bars
    
    if vicuna_f1_bars and mode == 'mean':
        # plot Vicuna baseline bar
        means_vicuna = [0.32, 0.38, 0.35, 0.30]
        stds_vicuna  = [0, 0, 0, 0]
        
        if len(categories) < 4: # if no DBpedia-WebNLG category (like with finetuned REBEL)
            means_vicuna.pop()
            stds_vicuna.pop()
    
        ax.bar(x, means_vicuna, width, label='Vicuna', yerr=[stds_vicuna], capsize=5, color="#000000")
    
    multiplier = 1
    colors = ["#800080", "#55308d", "#2a6099", "#158466", "#00a933", "#81d41a"]
    
    
    # llm_metric_folder_path is gpt-4o-4-shot for example
    # for every specified model metric folder, plot one bar for every variant
    for idx, (llm_metric_folder_path, plot_label) in enumerate(zip(metric_paths, plot_labels)):
        
        offset = width * multiplier
        
        if mode == 'mean':
            means, stds = get_means_stds(llm_metric_folder_path, metric)
            #print("bar", plot_label, means, stds, multiplier)
            ax.bar(x + offset, means, width, label=plot_label, yerr=[stds], capsize=5, color=colors[idx])
        elif mode == 'median':
            medians, p75errors, p25errors = get_median_percentiles(llm_metric_folder_path, metric)
            ax.bar(x + offset, medians, width, label=plot_label, yerr=[medians-p25errors, p75errors-medians], capsize=5, color=colors[idx])
        
        multiplier += 1
    
    plt.rcParams['figure.dpi'] = 600
    plt.rcParams["font.family"] = "sans-serif"
    plt.rcParams["font.sans-serif"] = ["DejaVu Serif"]

    if mode == 'mean':
        ax.set_title(f"{title_model_name}, Mean {metric} & Standard Deviation")
        ax.set_ylabel(f'Mean {metric.upper()}, standard deviation bars.')
    elif mode == 'median':
        ax.set_title(f"{title_model_name}, Median {metric} & p25 / p75 percentiles.")
        ax.set_ylabel(f'Median {metric.upper()}, percentile error bars.')
        
    ax.set_xlabel('Text2KGBench Variant')

    ax.set_xticks(x+width*2.5, categories)
    ax.legend(loc='upper left', ncols=3)
    ax.set_yticks(np.linspace(0, 1, 11))
    ax.set_ylim(0, ylim)
    
    filename = title_model_name.replace(',', '').replace(' ', '-')
    
    plt.savefig(f'../results/graphics/{filename}-{mode}-{metric}.png')
    

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Generate mean (or median) bar plots for specified models and metric with standard deviation (or p25 / p75 percentiles) errors.')
    parser.add_argument("--title_model_name", type=str, default='Default Title')
    parser.add_argument("--model_name_glob", type=str, default='')
    parser.add_argument("--model_names_list", type=str, nargs="*", default=[])
    parser.add_argument("--plot_labels", type=str, nargs="*", default=[])
    
    parser.add_argument("--mode", type=str, default='mean', choices=['mean', 'median'])
    parser.add_argument("--variants", type=str, nargs="*", default=["Unseen\nWikidata-TekGen", "Verified\nWikidata-TekGen", "All\nWikidata-TekGen", "All\nDBpedia-WebNLG"])
    parser.add_argument("--metric", type=str, default="avg_f1", choices=['avg_f1', 'avg_recall', 'avg_precision'])
    parser.add_argument("--ylim", type=float, default=1.0)
    parser.add_argument("--vicuna_f1_bars", type=str, choices=['True', 'False'])
    args = parser.parse_args()

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
        values = []
        print("plot_labels not provided, inferring from folder names...")
        a_folder_name = llm_metric_folders[0].split("/")[-1] 
        if 'beams' in a_folder_name:
            for path in llm_metric_folders:
                splitted_folder_name: list[str] = path.split("/")[-1].split("-")
                idx = splitted_folder_name.index("beams")
                plot_labels.append(f"{splitted_folder_name[idx-1]}-beams")
                values.append(float(splitted_folder_name[idx-1]))
            
        elif 'shot' in a_folder_name:
            for path in llm_metric_folders:
                splitted_folder_name: list[str] = path.split("/")[-1].split("-")
                idx = splitted_folder_name.index("shot")
                plot_labels.append(f"{splitted_folder_name[idx-1]}-shot")
                values.append(float(splitted_folder_name[idx-1]))
        
        elif 't=' in a_folder_name:
            for path in llm_metric_folders:
                tequals: str = path.split("/")[-1].split("-")[-1]
                plot_labels.append(f"{tequals[2:]}")
                values.append(float(tequals[2:]))
        
        else:
            print("Unable to infer from folder names, aborting...")
            exit(0)
            
        order = np.argsort(values)
        plot_labels = list(np.array(plot_labels)[order])
        llm_metric_folders = list(np.array(llm_metric_folders)[order])
        
    else:
        plot_labels = args.plot_labels
        
    
    print("Generating bar plots for metrics inside : ")
    pprint(llm_metric_folders)
    
    print("Using plot labels :", plot_labels)
    
    gen_bar_plots(llm_metric_folders, plot_labels, args.title_model_name, args.metric, args.variants, ylim=args.ylim, vicuna_f1_bars=strtobool(args.vicuna_f1_bars), mode=args.mode)
    
    # TODO : add parameter to generate median plots instead, shouldn't be that difficult,
    # modifiying line mean, stds = get_mean_stds(...) to medians, percentiles ought to do it
    # and make sure to update the titles and labels based on this too.
    # regenerate graphics with new llm_responses metrics, current graphs are based on older prompt without range/domain constraints.