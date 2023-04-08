import math 
import numpy as np
def generate_features(sequence_data,type_of_features) :
    # sequence_data : list of 10 last frames" body keypoints, track IDs and bboxes

    IDs_in_seq=find_all_IDs(sequence_data)
    people_features= {}
    for i in IDs_in_seq:
        people_features[i] = None
    for id in IDs_in_seq:
        person_sequence = []
        for frame in sequence_data:
            if type_of_features=='dist' :
                p_frame_features = np.zeros(24)
            else : 
                p_frame_features = np.zeros(28) 
            for p in frame :
                if p['track_id'] == id:
                    p_frame_features=features_of_person(p['keypoints'], type_of_features=type_of_features)
            person_sequence.append(p_frame_features)
        people_features[id] = person_sequence 
    return people_features
       
def features_of_person (p_kps, type_of_features="angl"):


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
    avrg_neck_x = (p_kps[neck_idx][0] + p_kps[neck_idx+1][0])/2.0
    avrg_neck_y = (p_kps[neck_idx][1] + p_kps[neck_idx+1][1])/2.0
    avrg_hip_x = (p_kps[hip_idx][0] + p_kps[hip_idx+1][0])/2.0
    avrg_hip_y = (p_kps[hip_idx][1] + p_kps[hip_idx+1][1])/2.0

    normalizator=np.sqrt((avrg_neck_x-avrg_hip_x)**2 + (avrg_neck_y-avrg_hip_y)**2)

    if type_of_features=="angl":
        p_features=[]
        for i in range(len(joints)):
            idx1 = joints[i][0]
            idx2 = joints[i][1]
            _, sin_val, cos_val = features_of_joint(p_kps[idx1][0],p_kps[idx1][1],p_kps[idx2][0],p_kps[idx2][1], normalizator)
            
            p_features.append(sin_val)
            p_features.append(cos_val)
    else:
        p_features=kps_dist_of_person (p_kps, normalizator)


    return p_features


def kps_dist_of_person (p_kps, normalizator):

    ref_kp=0 #nose

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



def features_of_joint(x1, y1, x2, y2, normalizator):

    dx = x2 - x1
    dy = y2 - y1

    angle = math.atan2(dy, dx)
    sin_val = math.sin(angle)
    cos_val = math.cos(angle)

    longitud=np.sqrt(dx**2 + dy**2)
    longitud=longitud/normalizator

    return longitud, sin_val, cos_val



def find_all_IDs(sequence_data):
    all_IDs=[]
    for frame in sequence_data:
        for p in frame:
            if p['track_id'] not in all_IDs:
                  all_IDs.append(p['track_id'])
    return all_IDs
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