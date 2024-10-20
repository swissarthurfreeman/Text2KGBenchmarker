import json
import glob
from pprint import pprint

train_files = glob.glob("./*.jsonlp")

for train_file in train_files:
    with open(train_file, "r") as sent_f:
        with open(train_file + "_folds", "r") as folds_f:
            folds = json.load(folds_f)
            pprint(folds)