import os
import sys
import cv2
import json

from ultralytics import YOLO

sys.path.insert(0, 'ViTPose_files')
from mmpose.apis import init_pose_model, inference_top_down_pose_model

'''
This script, extracts pose estimations into JSON files (one per frame) 
of all the videos in the folder {videos_path}
into the target folder {data_path}
'''

videos_path=r"A-Dataset-for-Automatic-Violence-Detection-in-Videos-master\violence-detection-dataset\non_violent\cam2"
data_path=r"result_peace_cam2"

# Initialize detection model
det_model = YOLO("yolov8n.pt")

# Initialize pose detection model
pose_model = init_pose_model("./configs/ViTPose_huge_simple_coco_256x192.py","./vitpose-h-simple.pth")


# For every video

for video_file_name in os.listdir(videos_path):
    video_file_path=os.path.join(videos_path,video_file_name)

    video_file_base = os.path.splitext(video_file_name)[0]
    data_file_path=os.path.join(data_path,video_file_base)

    if not os.path.exists(data_file_path):
        os.makedirs(data_file_path)

    results = det_model.track(source=video_file_path, tracker="bytetrack.yaml") 
    # Open the video file
    video_capture = cv2.VideoCapture(video_file_path)

    # Initialize a counter for the extracted frames
    frame_count = 0
    
    # Loop over the frames of the video
    while True:
        # Read the next frame from the video
        ret, frame = video_capture.read()

        # If there are no more frames, break out of the loop
        if not ret:
            break

        processed_result=results[frame_count].boxes.boxes.to('cpu').numpy()
        person_results=[]
        for p in range(len(processed_result)):
            person = {}
            person['track_id'] = int(processed_result[p][4])
            processed_result[p][4] = processed_result[p][5]
            person['bbox'] = processed_result[p][range(5)]
            person_results.append(person)

        out, _ = inference_top_down_pose_model(pose_model,frame,person_results,bbox_thr=0.1,format='xyxy')

        for i in range(len(out)):
            out[i]['bbox']=out[i]['bbox'].tolist()
            out[i]['keypoints']=out[i]['keypoints'].tolist()

        # Construct the output file path for the current frame
        result_file_path=os.path.join(data_file_path,f"{(frame_count+1):03d}.json")
        # Open the file in write mode and write the data to it
        with open(result_file_path, "w") as file:
            # Serialize the list to JSON format
            json.dump(out, file,indent=2)

        frame_count += 1
        

    # Release the video capture object
    video_capture.release()

