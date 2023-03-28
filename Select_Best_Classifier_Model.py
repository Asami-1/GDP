import os
from glob import glob
import pandas as pd
import json

# Folder containing all JSON files
WEIGHTS_DIRECTORY_PATH = 'weights/'

# Attribute based on which best model weights are to be selected
SELECTION_KEY = "best_test_acc"

# Min or Max value of SELECTION_KEY to decide the best model
SELECT_MAX = True

def read_all_json_files(weights_directory):
    weights_path = weights_directory
    json_files = glob(os.path.join(weights_path,"**/*.json"), recursive=True)
    data = []
    for file in json_files:
        with open(file, 'r') as f:
            json_data = json.load(f)
            json_data['file_path'] = os.path.abspath(file)
            data.append(json_data)
    df = pd.DataFrame(data)
    return df

def get_best_model_weights_path(weights_directory, selection_criteria, select_max = True):
    df = read_all_json_files(weights_directory)
    if select_max:
        best = df.loc[df[selection_criteria].idxmax()]
    else:
        best = df.loc[df[selection_criteria].idxmin()]
    best_wt = os.path.splitext(best['file_path'])[0] + '.h5'
    print('Best model path among all combinations is:')
    return best_wt


best_model_wt_path = get_best_model_weights_path(WEIGHTS_DIRECTORY_PATH, SELECTION_KEY, SELECT_MAX)

best_model_wt_path
