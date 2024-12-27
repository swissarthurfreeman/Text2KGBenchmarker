import json

for i in [1, 2, 3, 4, 5, 6]:
    model_name = f'gpt-4o-{i}-shot'

    with open(model_name + "/dpedia_webnlg_clean_avg.jsonl") as f:
        data_dbpedia = [json.loads(line) for line in f]
        
        with open(model_name + "/dbpedia_webnlg_clean_avg_per_ontology.csv", "a") as f:
            f.write("onto, P, R, F1, OC, SH, RH, OH\n")
            for ont_result in data_dbpedia:    
                f.write(f"{ont_result['onto']}, {ont_result['all']['avg_precision']:.2f}, {ont_result['all']['avg_recall']:.2f}, {ont_result['all']['avg_f1']:.2f}, ")
                f.write(f"{ont_result['all']['avg_onto_conf']:.2f}, {ont_result['all']['avg_sub_halluc']:.2f}, {ont_result['all']['avg_rel_halluc']:.2f}, {ont_result['all']['avg_obj_halluc']:.2f}\n")
            
    with open(model_name + "/wikidata_tekgen_avg.jsonl") as f:
        data_wikidata = [json.loads(line) for line in f]
        for variant in ["all", "unseen", "verified"]:
            with open(model_name + f"/wikidata_tekgen_avg_per_ontology_{variant}.csv", "a") as f:
                f.write("onto, P, R, F1, OC, SH, RH, OH\n")
                for ont_result in data_wikidata:
                    f.write(f"{ont_result['onto']}, {ont_result[variant]['avg_precision']:.2f}, {ont_result[variant]['avg_recall']:.2f}, {ont_result[variant]['avg_f1']:.2f}, ")
                    f.write(f"{ont_result[variant]['avg_onto_conf']:.2f}, {ont_result[variant]['avg_sub_halluc']:.2f}, {ont_result[variant]['avg_rel_halluc']:.2f}, {ont_result[variant]['avg_obj_halluc']:.2f}\n")
        