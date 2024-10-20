import glob
import json
from pprint import pprint

files = glob.glob("./*.jsonl")

print(files)

for train_sent_file_path in files:

    with open(train_sent_file_path, "r") as sent_f:
        
        folds_dic = {}  # {removed_sent_id: kept_sent_id}
        dic = {}
        data = [json.loads(sent) for sent in sent_f]
        
        for sent_json in data:
            if sent_json["sent"] in dic.keys():
                folds_dic[sent_json["id"]] = dic[sent_json["sent"]]["id"]
                print("fold :", sent_json["id"], " to : ", dic[sent_json["sent"]]["id"])
                
                dic[sent_json["sent"]]["unseen"] = dic[sent_json["sent"]]["unseen"] or sent_json["unseen"] 
                dic[sent_json["sent"]]["verified"] = dic[sent_json["sent"]]["verified"] or sent_json["verified"]
                
                dic[sent_json["sent"]]["triples"] += sent_json["triples"]
            else:
                dic[sent_json["sent"]] = sent_json 
        
    #pprint(dic)
    res = dic.values()
    
    with open(train_sent_file_path + "p", "w") as sent_res_f:
        for val in res:
            sent_res_f.write(json.dumps(val) + "\n")
            
    with open(train_sent_file_path + "p_folds", "w") as sent_folds_f:
        sent_folds_f.write(json.dumps(folds_dic))
                    
