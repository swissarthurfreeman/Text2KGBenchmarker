# proof that re_score does the same thing as our implementation, if triples are normalized before hand.

import numpy as np

def re_score(gt_relations, pred_relations, relation_types, mode="boundaries"):
    """Evaluate RE predictions
    Args:
        pred_relations (list) :  list of list of predicted relations (several relations in each sentence)
        gt_relations (list) :    list of list of ground truth relations
            rel = { "head": (start_idx (inclusive), end_idx (exclusive)),
                    "tail": (start_idx (inclusive), end_idx (exclusive)),
                    "head_type": ent_type,
                    "tail_type": ent_type,
                    "type": rel_type}
        vocab (Vocab) :         dataset vocabulary
        mode (str) :            in 'strict' or 'boundaries' """

    assert mode in ["strict", "boundaries"]
    
    #print("Computing re_score, len(pred_relations), len(gt_relations)", len(pred_relations), len(gt_relations))
    #print("\n\n pred_relations[0] len\n", pred_relations[0], len(pred_relations[0]), "\n gt_relations[0], len", gt_relations[0], len(gt_relations[0]))
    #print("relations", relation_types)
    #print("\n\n")
    # relation_types = [v for v in relation_types if not v == "None"]
    scores = {rel: {"tp": 0, "fp": 0, "fn": 0} for rel in relation_types + ["ALL"]}

    # Count GT relations and Predicted relations
    n_sents = len(gt_relations)
    n_rels = sum([len([rel for rel in sent]) for sent in gt_relations])
    
    # BUG : multi beam concatenation is hurting precision, since the metric here doesn't merge triples... 
    n_found = sum([len([rel for rel in sent]) for sent in pred_relations])

    # Count TP, FP and FN per type
    for pred_sent, gt_sent in zip(pred_relations, gt_relations):
        for rel_type in relation_types:
            # strict mode takes argument types into account
            if mode == "strict":
                pred_rels = {(rel["head"], rel["head_type"], rel["tail"], rel["tail_type"]) for rel in pred_sent if
                             rel["type"] == rel_type}
                gt_rels = {(rel["head"], rel["head_type"], rel["tail"], rel["tail_type"]) for rel in gt_sent if
                           rel["type"] == rel_type}

            # boundaries mode only takes argument spans into account
            elif mode == "boundaries":
                #print("pred_sent", pred_sent)
                pred_rels = {(rel["head"], rel["tail"]) for rel in pred_sent if rel["type"] == rel_type}
                gt_rels = {(rel["head"], rel["tail"]) for rel in gt_sent if rel["type"] == rel_type}

            scores[rel_type]["tp"] += len(pred_rels & gt_rels)
            scores[rel_type]["fp"] += len(pred_rels - gt_rels)
            scores[rel_type]["fn"] += len(gt_rels - pred_rels)

    # Compute per relation Precision / Recall / F1
    for rel_type in scores.keys():
        if scores[rel_type]["tp"]:
            scores[rel_type]["p"] = 100 * scores[rel_type]["tp"] / (scores[rel_type]["fp"] + scores[rel_type]["tp"])
            scores[rel_type]["r"] = 100 * scores[rel_type]["tp"] / (scores[rel_type]["fn"] + scores[rel_type]["tp"])
        else:
            scores[rel_type]["p"], scores[rel_type]["r"] = 0, 0

        if not scores[rel_type]["p"] + scores[rel_type]["r"] == 0:
            scores[rel_type]["f1"] = 2 * scores[rel_type]["p"] * scores[rel_type]["r"] / (
                    scores[rel_type]["p"] + scores[rel_type]["r"])
        else:
            scores[rel_type]["f1"] = 0

    # Compute micro F1 Scores
    tp = sum([scores[rel_type]["tp"] for rel_type in relation_types])
    fp = sum([scores[rel_type]["fp"] for rel_type in relation_types])
    fn = sum([scores[rel_type]["fn"] for rel_type in relation_types])

    if tp:
        precision = 100 * tp / (tp + fp)
        recall = 100 * tp / (tp + fn)
        f1 = 2 * precision * recall / (precision + recall)

    else:
        precision, recall, f1 = 0, 0, 0

    scores["ALL"]["p"] = precision
    scores["ALL"]["r"] = recall
    scores["ALL"]["f1"] = f1
    scores["ALL"]["tp"] = tp
    scores["ALL"]["fp"] = fp
    scores["ALL"]["fn"] = fn

    # Compute Macro F1 Scores
    scores["ALL"]["Macro_f1"] = np.mean([scores[ent_type]["f1"] for ent_type in relation_types])
    scores["ALL"]["Macro_p"] = np.mean([scores[ent_type]["p"] for ent_type in relation_types])
    scores["ALL"]["Macro_r"] = np.mean([scores[ent_type]["r"] for ent_type in relation_types])

    print(f"RE Evaluation in *** {mode.upper()} *** mode")

    print(
        "processed {} sentences with {} relations; found: {} relations; correct: {}.".format(n_sents, n_rels, n_found,
                                                                                             tp))
    print(
        "\tALL\t TP: {};\tFP: {};\tFN: {}".format(
            scores["ALL"]["tp"],
            scores["ALL"]["fp"],
            scores["ALL"]["fn"]))
    print(
        "\t\t(m avg): precision: {:.2f};\trecall: {:.2f};\tf1: {:.2f} (micro)".format(
            precision,
            recall,
            f1))
    print(
        "\t\t(M avg): precision: {:.2f};\trecall: {:.2f};\tf1: {:.2f} (Macro)\n".format(
            scores["ALL"]["Macro_p"],
            scores["ALL"]["Macro_r"],
            scores["ALL"]["Macro_f1"]))

    for rel_type in relation_types:
        print("\t{}: \t\t\tTP: {};\tFP: {};\tFN: {};\tprecision: {:.2f};\trecall: {:.2f};\tf1: {:.2f};\t{}".format(
            rel_type,
            scores[rel_type]["tp"],
            scores[rel_type]["fp"],
            scores[rel_type]["fn"],
            scores[rel_type]["p"],
            scores[rel_type]["r"],
            scores[rel_type]["f1"],
            scores[rel_type]["tp"] +
            scores[rel_type]["fp"]))

    return scores, precision, recall, f1



gt_triples = [
	{"head": "Spirited Away", "type": "genre", "tail": "Fantasy film"},
	{"head": "Spirited Away", "type": "director", "tail": "Hayao Miyazaki"},
	{"head": "Spirited Away", "type": "publication date", "tail": "01 January 2001"},
	{"head": "Spirited Away", "type": "production company", "tail": "Studio Ghibli"},
	{"head": "Spirited Away", "type": "screenwriter", "tail": "Hayao Miyazaki"}
]

llm_triples = [
	{"head": "Spirited Away", "type": "director", "tail": "Hayao Miyazaki"},
	{"head": "Spirited Away", "type": "genre", "tail": "Fantasy film"},
	{"head": "Spirited Away", "type": "production company", "tail": "Tokuma Shoten"}
]

for triple in gt_triples:
	triple["head"] = triple["head"].lower()
	triple["tail"] = triple["tail"].lower()
	triple["type"] = triple["type"].lower()

for triple in llm_triples:
	triple["head"] = triple["head"].lower()
	triple["tail"] = triple["tail"].lower()
	triple["type"] = triple["type"].lower()




re_score([gt_triples], [llm_triples], ["genre", "director", "publication date", "production company", "screenwriter"])









