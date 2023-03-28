# Imports
import os
import numpy as np
import math
import json
import shutil
import random
from json.decoder import JSONDecodeError


# Find all IDs in a JSON list
def find_all_IDs(path_to_folder_with_json, JSON_list, seq_idx):
    all_IDs=[]
    for idx, json_frame in enumerate(JSON_list):
            if idx in seq_idx:
                frame_path=os.path.join(path_to_folder_with_json,json_frame)
                data=[]
                with open(frame_path, 'r') as f:
                        try:
                            data = json.load(f)
                            for p in data:
                                if p['track_id'] not in all_IDs:
                                    all_IDs.append(p['track_id'])
                        except JSONDecodeError:
                            pass
    return all_IDs

def most_common_class(numbers):
    counts = {}
    for num in numbers:
        if num in counts:
            counts[num] += 1
        else:
            counts[num] = 1

    max_count = max(counts.values())
    max_nums = [num for num, count in counts.items() if count == max_count]

    if len(max_nums) == 1:
        return max(counts, key=counts.get)
    else:
        if -1 in max_nums:
            return -1
        elif 1 in max_nums:
            return 1
        elif 2 in max_nums:
            return 2
# Extract features of person in a frame

def features_of_joint(x1, y1, x2, y2, normalizator):
    dx = x2 - x1
    dy = y2 - y1

    angle = math.atan2(dy, dx)
    sin_val = math.sin(angle)
    cos_val = math.cos(angle)

    longitud=np.sqrt(dx**2 + dy**2)
    longitud=longitud/normalizator

    return longitud, sin_val, cos_val

def kps_dist_of_person (p_kps, normalizator):

    ref_kp_x=p_kps[ref_kp][0]
    ref_kp_y=p_kps[ref_kp][1]

    p_features=[]
    for kp in range(len(p_kps)):
        if kp >= 5:
            dx = p_kps[kp][0] - ref_kp_x
            dy = p_kps[kp][1] - ref_kp_y

            p_features.append(dx/normalizator)
            p_features.append(dy/normalizator)
   
    return p_features


def features_of_person (p_kps, type_of_features="long_angl"):

    avrg_neck_x = (p_kps[neck_idx][0] + p_kps[neck_idx+1][0])/2.0
    avrg_neck_y = (p_kps[neck_idx][1] + p_kps[neck_idx+1][1])/2.0
    avrg_hip_x = (p_kps[hip_idx][0] + p_kps[hip_idx+1][0])/2.0
    avrg_hip_y = (p_kps[hip_idx][1] + p_kps[hip_idx+1][1])/2.0

    normalizator=np.sqrt((avrg_neck_x-avrg_hip_x)**2 + (avrg_neck_y-avrg_hip_y)**2)

    if type_of_features=="long_angl" or type_of_features=="angl":
        p_features=[]
        for i in range(len(joints)):
            idx1 = joints[i][0]
            idx2 = joints[i][1]
            longitud, sin_val, cos_val = features_of_joint(p_kps[idx1][0],p_kps[idx1][1],p_kps[idx2][0],p_kps[idx2][1], normalizator)
            if type_of_features=="long_angl":
                p_features.append(longitud)
            p_features.append(sin_val)
            p_features.append(cos_val)
    else:
        p_features=kps_dist_of_person (p_kps, normalizator)


    return p_features

# Extract seq_from_video

def extract_seq_from_video(jsons_folder_path, output_seq_path, classes_counters, seq_duration=1, seq_fps=10, window_seconds=0.5, 
                           video_fps=30, type_of_features="angl"):

    video_pose_name=os.path.split(jsons_folder_path)[1]
    video_name=video_pose_name[0:(len(video_pose_name)-5)]  

    if type_of_features=="long_angl":
        n_features_per_person=3*len(joints) 
    elif type_of_features=="angl":
        n_features_per_person=2*len(joints)
    else:
        n_features_per_person=2*(12)


    jsons_list=os.listdir(jsons_folder_path)
    jsons_list.sort()

    finaljsons_per_seq=seq_duration*seq_fps
    videoframes_per_seq=seq_duration*video_fps
    window=math.floor(video_fps*window_seconds)
    one_frame_every=video_fps//seq_fps

    frames_in_video=len(jsons_list)-1
    n_seq=1+math.ceil((frames_in_video-videoframes_per_seq)/window)
    
    for seq in range(n_seq):

        videoframe_start=1+seq*window
        videoframe_finish=videoframe_start+videoframes_per_seq
        framevideo_idx=list(range(videoframe_start,videoframe_finish))
        every_framevideo_idx=[]
        for i in framevideo_idx:
            if (i-1)%one_frame_every ==0:
                every_framevideo_idx.append(i)
        IDs_in_seq=find_all_IDs(jsons_folder_path, jsons_list, every_framevideo_idx)
        
        for id in IDs_in_seq:
            classes_of_ID_in_seq=[]
            for frame_idx in every_framevideo_idx:
                class_id_frame=-1
                if frame_idx+1<=frames_in_video:
                    frame_path=os.path.join(jsons_folder_path,jsons_list[frame_idx])
                    with open(frame_path, 'r') as f:
                        try:
                            jsonframe_data = json.load(f)
                            for p in jsonframe_data:
                                if p['track_id']==id:
                                    class_id_frame=p['class']
                                    break
                        except JSONDecodeError:
                            pass
                classes_of_ID_in_seq.append(class_id_frame)
            seq_class=most_common_class(classes_of_ID_in_seq)
            if seq_class != -1:
                features_of_ID_in_seq=[]
                for frame_idx in every_framevideo_idx:
                    p_frame_features=np.zeros(n_features_per_person).tolist()
                    if frame_idx+1<=frames_in_video:    
                        frame_path=os.path.join(jsons_folder_path,jsons_list[frame_idx])
                        with open(frame_path, 'r') as f:
                            try:
                                jsonframe_data = json.load(f) 
                                for p in jsonframe_data:
                                    if p['track_id']==id:
                                        p_frame_features=features_of_person(p['keypoints'], type_of_features=type_of_features)
                            except JSONDecodeError:
                                pass
                    features_of_ID_in_seq.append(p_frame_features)

                data_id_seq={}
                data_id_seq['ID']=id
                data_id_seq['class']=seq_class
                data_id_seq['features']=features_of_ID_in_seq
                
                with open(os.path.join(output_seq_path,f"{video_name}_seq_{seq+1}_id_{id}.json"), "w") as f:
                    # Serialize the list to JSON format
                    json.dump(data_id_seq, f, indent=2)

                if seq_class == 0:
                    classes_counters['neutral']+=1
                elif seq_class == 1:
                    classes_counters['aggressive']+=1
                else:
                    classes_counters['victim']+=1
                  
    return classes_counters
# Generate all seqs

#seq_fps = 30,15,10,6,5,3,2,1 (default: 10)

def generate_all_seqs(data_path, split_data_path, args):

    if args['use_peaceful']:
        sub_data=["aggressive","peaceful"]
        sub_idx="a+p"
    else:
        sub_data=["aggressive"]
        sub_idx="a"

    if not os.path.exists(split_data_path):
        os.makedirs(split_data_path)
        number_of_runs=1
    else:
        number_of_runs=1+len(os.listdir(split_data_path))

    run_name="run"+str(number_of_runs)+"_"+sub_idx+"_"+args['type_of_features']+"_"+str(args['seq_duration'])+"_"+str(args['window_seconds'])
    output_train_path=os.path.join(split_data_path,run_name,"train")
    if not os.path.exists(output_train_path):
        os.makedirs(output_train_path)
    output_test_path=os.path.join(split_data_path,run_name,"test")
    if not os.path.exists(output_test_path):
        os.makedirs(output_test_path)

    train_classes_counters={'neutral':0, 'aggressive':0, 'victim':0}
    test_classes_counters={'neutral':0, 'aggressive':0, 'victim':0}
    for sub in sub_data:
        subdata_path=os.path.join(data_path,sub)    

        videos_list=os.listdir(subdata_path)
        
        # Partition the number of videos
        num_videos = len(videos_list)
        num_videos_train = int(num_videos * args['train_split_percent'] / 100)

        # Randomly shuffle video files list
        videos_list.sort()
        random.seed(5) 
        random.shuffle(videos_list)

        for i in range(num_videos_train):
            jsons_folder_path=os.path.join(subdata_path,videos_list[i],videos_list[i]+"_pose")
            
            train_classes_counters=extract_seq_from_video(jsons_folder_path, output_train_path, train_classes_counters,
                                                          seq_duration=args['seq_duration'],seq_fps=10, 
                                                          window_seconds=args['window_seconds'], video_fps=30, 
                                                          type_of_features=args['type_of_features'])
            
        for i in range(num_videos_train, num_videos):
            jsons_folder_path=os.path.join(subdata_path,videos_list[i],videos_list[i]+"_pose")
            
            test_classes_counters=extract_seq_from_video(jsons_folder_path, output_test_path, test_classes_counters,
                                                         seq_duration=args['seq_duration'], seq_fps=10, 
                                                         window_seconds=args['window_seconds'], video_fps=30, 
                                                         type_of_features=args['type_of_features'])
    train_per=args['train_split_percent']
    print("All sequences have been generated and splited in ",train_per,"-",100-train_per,"%:")
    print("   Train =>",train_classes_counters)
    print("   Test  =>",test_classes_counters)




#args['seq_duration'] = 1 #in seconds
#args['type_of_features'] = "dist" #long_angl, angl, or dist
#args['window_seconds'] = 0.5 #in seconds
#args['use_peaceful'] = True
#args['train_split_percent'] = 80 #%


#generate_all_seqs(data_path, split_data_path, args)



if __name__ == "__main__":

    # Some variables
    GDP_path=os.getcwd()

    data_path=os.path.join(GDP_path, "data")
    split_data_path=os.path.join(GDP_path, "split_seqs")

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

    neck_idx=5
    hip_idx=11

    ref_kp=0 #nose

    args={}
    args['seq_duration'] = int(input("Insert sequence duration (in seconds): "))
    args['type_of_features'] = input("Insert type of features to extract (dist or angl): ") #long_angl, angl, or dist
    args['window_seconds'] = float(input("Insert sequence window (in seconds): "))
    use_data = int(input("Insert 1 to use peaceful data, or 0 to not use it: "))
    if use_data==0:
        args['use_peaceful'] = False
    else:
        args['use_peaceful'] = True
    args['train_split_percent'] = int(input("Insert train split percentage (in %): "))

    generate_all_seqs(data_path, split_data_path, args)