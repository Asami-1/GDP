import os 
import json

'''
This script iterates over the json of each video folder in data_full and assigns 
low track ids (starting from 0) to each person on the video.

'''
def replace_track_id(filepath, lowest_track_id):
    with open(filepath, 'r') as f:
        data = json.load(f)

    for person_dict in data:
        person_dict['track_id'] -= lowest_track_id

    with open(filepath, 'w') as f:
        json.dump(data, f,indent=4)

def get_track_id(filepath):
    with open(filepath, 'r') as f:
        data = json.load(f)
        return data[0]['track_id']

data_paths = ["data_full/train/peaceful","data_full/test/peaceful","data_full/train/conflict","data_full/train/peaceful"]
for data_folder in data_paths:
    print(data_folder)
    for data_point in os.listdir(data_folder):
        print(data_point)
        for json_path in os.listdir(data_folder+'/'+data_point) :
            if json_path == '001.json':
                lowest_track_id = get_track_id(data_folder+'/'+data_point+'/'+json_path)
                print(lowest_track_id)
            replace_track_id(data_folder+'/'+data_point+'/'+json_path,lowest_track_id)




