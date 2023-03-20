from flask import Flask, request,Response, jsonify, render_template
import os
import io
import numpy as np
import PIL.Image as Image
import matplotlib.pyplot as plt
from ultralytics import YOLO
from deep_sort_realtime.deepsort_tracker import DeepSort
from custom_tracking import custom_tracking
import cv2
import time
from mmpose.apis import (inference_top_down_pose_model, init_pose_model,
                         process_mmdet_results, vis_pose_result)

det_model = YOLO("yolov8n.pt")

pose_config = "./configs/ViTPose_small_coco_256x192.py"
pose_checkpoint = "./vitpose_small.pth"
pose_model = None
tracker = DeepSort(max_age=5)

app = Flask(__name__)


frames = [None]
images = [None]

def load_models():
    global det_model, pose_model
    pose_model = init_pose_model(pose_config,pose_checkpoint)


def YOLO_preprocess(bbs_YOLO):
    bbs=[]
    for p, p_data in enumerate(bbs_YOLO):
        person=[]
        person.append(bbs_YOLO[p][range(4)])
        person.append(bbs_YOLO[p][4])
        person.append(bbs_YOLO[p][5])
        bbs.append(person)
    
    return bbs


def process_yolo_results (yolo_results):

    results=yolo_results[0].boxes.boxes.to('cpu').numpy()

    person_results=[]
    for i in range(len(results)):
        person = {}
        person['bbox'] = results[i][range(5)]
        person_results.append(person)

    return person_results

def DS_preprocess(tracks):
    person_results=[]
    for i in range(len(tracks)):
        person = {}
        person['bbox'] = np.append(tracks[i].to_ltwh(),tracks[i].det_conf)
        person['track_id'] = tracks[i].track_id
        person_results.append(person)

    return person_results

def draw_data_on_image(image,results) :
    copy = np.copy(image)
    for detection in results:
        # Get the bounding box coordinates
        bbox = detection['bbox']
        x1, y1, x2, y2 = map(int, bbox[:4])

        # Draw the bounding box
        cv2.rectangle(copy, (x1, y1), (x2, y2), (0, 255, 0), 2)

        # Get the body keypoints
        keypoints = detection['keypoints']

        # Loop over the keypoints and draw them
        for point in keypoints:
            x, y, _ = map(int, point)
            cv2.circle(copy, (x, y), 3, (0, 0, 255), -1)
    
    return copy

def gen():  

    while True :
        start_yolo = time.time()
        image = np.array(Image.open(io.BytesIO(frame)))

        if not (image == images[-1]).all() :
            images.append(image)
            # -------------  YOLO-------------
            # print("added a frame to frame list")
            # print('current number of frames :',len(images))            
            results = det_model(image,classes=0,conf=0.4,iou=0.6)
            bbs=YOLO_preprocess(results[0].boxes.boxes.cpu().numpy())
           
            end_yolo = time.time()
            print("Yolo detection time :",end_yolo-start_yolo) 

        #  ------------------------ DeepSort -------------------------------
            start_DS = time.time()
            tracks = tracker.update_tracks(bbs, frame=image) # bbs expected to be a list of detections, each in tuples of ( [left,top,w,h], confidence, detection_class )
            
            # Find indexes to delete 
            idx_to_delete=[]
            for p in range(len(tracks)):
                found=False
                for p2 in range(len(bbs)):
                    if (tracks[p].to_ltwh(orig=True) == bbs[p2][0]).all():
                        found=True
                if not found:
                    idx_to_delete.append(p)
            idx_to_delete.reverse()
            for idx in idx_to_delete:
                del tracks[idx]                   
            end_DS = time.time()
            print("Deep Sort tracking time :",end_DS-start_DS) 

            start_VitPose = time.time()
            person_results = DS_preprocess(tracks)

            out, _ = inference_top_down_pose_model(pose_model,image,person_results,bbox_thr=0.4,format='xyxy')
            for j in range(len(out)):
                out[j]['bbox']=out[j]['bbox'].tolist()
                out[j]['keypoints']=out[j]['keypoints'].tolist()

            end_VitPose= time.time()

            print("VitPose estimation time :",end_VitPose-start_VitPose) 
            print('TOTAL TIME : ',end_VitPose - start_yolo)
            print("\n------------------------------------\nresults of VitPose : ", [outs['track_id'] for outs in out],"\n")


            # Update VitPose outputs with track IDs
            for i,track in enumerate(tracks):
                if not track.is_confirmed():
                    continue
                out[i]['track_id'] = track.track_id

            image_with_output = draw_data_on_image(image,out)
            _, frame_with_output = cv2.imencode('.jpg', image_with_output)

            yield (b'--frame\r\nContent-Type: image/jpeg\r\n\r\n' + frame_with_output.tostring() + b'\r\n')







@app.route('/upload', methods=['PUT'])
def upload():
    global frame

    # keep jpg data in global variable
    frame = request.data
    
    return "OK"


@app.route('/video')
def video():      
    if frame:
        # if you use `boundary=other_name` then you have to yield `b--other_name\r\n`
        return Response(gen(), mimetype='multipart/x-mixed-replace; boundary=frame')
    else:
        return ""


@app.route('/', methods=['GET'])
def home():
    return render_template("index.html")


@app.route('/classify', methods=['GET'])
def classify():
    # Classifiy whether there is violent behaviour or not
    pass

def get_keypoints():
    # Get frames
    # Apply VitPose to get body keypoints
    pass


if __name__ == "__main__":
    load_models()
    port = int(os.environ.get('PORT', 5000))
    app.run(debug=False, host='0.0.0.0', port=port)