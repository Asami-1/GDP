import os
import numpy as np
import math
import json

# Joints (pairs of connected keypoints). ViTPose skeleton
joints=[[0,5],
        [0,6],
        [5,6],
        [5,7],
        [7,9],
        [6,8],
        [8,10],
        [11,12],
        [5,11],
        [6,12],
        [11,13],
        [13,15],
        [12,14],
        [14,16],
        ]

neck_idx=5  # These indexes are used to normalize joint longitud
hip_idx=11

def computeFeatures(x1, y1, x2, y2, normalizator):
    dx = x2 - x1
    dy = y2 - y1

    angle = math.atan2(dy, dx)
    sin_val = math.sin(angle)
    cos_val = math.cos(angle)

    longitud=np.sqrt(dx**2 + dy**2)
    longitud=longitud/normalizator

    return longitud, sin_val, cos_val


def extractFeatures (keypoints_data):

    longituds = []
    sin_vals = []
    cos_vals = []

    avrg_neck_x = (keypoints_data[neck_idx][0] + keypoints_data[neck_idx+1][0])/2.0
    avrg_neck_y = (keypoints_data[neck_idx][1] + keypoints_data[neck_idx+1][1])/2.0
    avrg_hip_x = (keypoints_data[hip_idx][0] + keypoints_data[hip_idx+1][0])/2.0
    avrg_hip_y = (keypoints_data[hip_idx][1] + keypoints_data[hip_idx+1][1])/2.0

    normalizator=np.sqrt((avrg_neck_x-avrg_hip_x)**2 + (avrg_neck_y-avrg_hip_y)**2)

    for i in range(len(joints)):
        id1 = joints[i][0]
        id2 = joints[i][1]
        longitud, sin_val, cos_val = computeFeatures(keypoints_data[id1][0],keypoints_data[id1][1],keypoints_data[id2][0],keypoints_data[id2][1], normalizator)
        longituds.append(longitud)
        sin_vals.append(sin_val)
        cos_vals.append(cos_val)

    row_features=[]
    for i in range(len(joints)):
        row_features.append(longituds[i])
        row_features.append(sin_vals[i])
        row_features.append(cos_vals[i])
        
    
    return row_features

def features_from_data_folder(root_folder_path, seq_results_path, frames_per_seq, window):
    '''
    root_folder_path: Path to root folder containing subfolders with JSON files with ViTPose results
    seq_results_path: Target folder
    frames_per_seq
    window: (Integer < frames_per_seq). At every "window" frames a sequence starts.
            By doing window = frames_per_seq => sequence only starts after previous sequence ends
    '''
    n_seq = 1
    for root, dirs, files in os.walk(root_folder_path):
        for dir_name in dirs:
            folder_path = os.path.join(root, dir_name)
            folder_name = os.path.split(folder_path)[1]
            json_files = [f for f in os.listdir(folder_path) if f.endswith('.json')]

            frames_in_video = len(json_files)
            starting_seq_frame = list(range(0, frames_in_video - frames_per_seq + 1, window))

            for frame_counter_base in starting_seq_frame:
                seq_features = [[] for i in range(frames_per_seq)]
                for frame in range(frames_per_seq):
                    frame_counter = frame_counter_base + frame
                    json_path = os.path.join(folder_path, json_files[frame_counter])

                    with open(json_path, 'r') as f:
                        frame_data = json.load(f)

                    max_people = 5 

                    frame_features = np.full((max_people, 3 * len(joints)), 0.0) # If there is less than 5 people in the frame, sets the keypoints to 0

                    for i in range(len(frame_data)):
                        row = frame_data[i]['track_id']
                        if row >= max_people:
                            print(f"Error: row {folder_name} is causing the problem")
                            continue
                        frame_features[row] = extractFeatures(frame_data[i]['keypoints'])

                    seq_features[frame] = frame_features.tolist()

                if not os.path.exists(seq_results_path):
                    os.makedirs(seq_results_path)

                with open(os.path.join(seq_results_path, f"{folder_name}_seq_{n_seq}.json"), "w") as file:
                    # Serialize the list to JSON format
                    json.dump(seq_features, file, indent=2)

                n_seq += 1
                
""" 

# Use:

# features_from_data_folder(root_folder_path, seq_results_path, frames_per_seq, window)


features_from_data_folder("/Users/irem/Desktop/data_full_deleted/train/peaceful","/Users/irem/Desktop/sample/train/Peaceful",30,30)

features_from_data_folder("/Users/irem/Desktop/data_full_deleted/test/peaceful","/Users/irem/Desktop/sample/test/Peaceful",30,30)

features_from_data_folder("/Users/irem/Desktop/data_full_deleted/train/conflict","/Users/irem/Desktop/sample/train/Agressive",30,30)

features_from_data_folder("/Users/irem/Desktop/data_full_deleted/test/conflict","/Users/irem/Desktop/sample/test/Agressive",30,30)


"""
