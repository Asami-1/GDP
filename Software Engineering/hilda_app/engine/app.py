from flask import Flask, request, Response, jsonify, render_template
from confluent_kafka import Producer
import os
import io
import json
import numpy as np
import PIL.Image as Image
import matplotlib.pyplot as plt
from ultralytics import YOLO
from deep_sort_realtime.deepsort_tracker import DeepSort
import cv2
import random
from mmpose.apis import (inference_top_down_pose_model, init_pose_model)
from LSTM_preprocess_app import generate_features
from tensorflow.keras.models import load_model 
import logging 
import time 

pose_config = "./configs/ViTPose_small_coco_256x192.py"
pose_checkpoint = "./weights/vitpose_small.pth"


### Build the constructs for saving the message to a log file
logging.basicConfig(format='%(asctime)s %(message)s',
                    datefmt='%Y-%m-%d %H:%M:%S',
                    filename='hildaprod.log',
                    filemode='w')

logger = logging.getLogger()
logger.setLevel(logging.INFO)

## Instantiate the connection to HILDA KAFKA Cluster - this connection should only be used for the purpose of this project from with>
try:
    p = Producer({'bootstrap.servers' : 'udr.dartec.cranfield.ac.uk:9092'})
    print('Kafka producer has been initialized...')
except : 
    print('Kafka failed to initialize'
          )



def receipt(err,msg):
    if err is not None:
        print('Error: {}'.format(err))
    else:
        message = 'HILDA has Produced message on topic {} with value of {}\n'.format(msg.topic(), msg.value().decode('utf-8'))
        logger.info(message)
        print(message)

app = Flask(__name__)
event_id = 1

frames = [None]
images = [None]

def load_models():
    global det_model, pose_model, classifier, tracker
    classifier =  load_model("./weights/BI_LSTM")
    det_model = YOLO("./weights/yolov8n.pt")
    tracker = DeepSort(max_age=15)
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

def draw_data_on_image(image,results,labels) :
    copy = np.copy(image)
    colors = {'0':(0,255,0),
     '1':(0,0,255) ,
     '2':(255,0,0)}
    for detection in results : 
        (f'labels {labels} \n detection : {detection}')
        # Get the bounding box coordinates
        bbox = detection['bbox']
        x1, y1, x2, y2 = map(int, bbox[:4])

        # Draw the bounding box
        if labels and detection['track_id'] in labels.keys() :
            cv2.rectangle(copy, (x1, y1), (x2, y2), colors[str(labels[detection['track_id']])], 2)
            # copy = cv2.putText(copy, str(detection['track_id']), (int(detection['bbox'][0]), int(detection['bbox'][1])), cv2.FONT_HERSHEY_SIMPLEX, 0.005*min(detection['bbox'][2],detection['bbox'][3]), colors[str(labels[detection['track_id']])], 5)
        else:
            cv2.rectangle(copy, (x1, y1), (x2, y2), colors['0'], 2)
            # copy = cv2.putText(copy, str(detection['track_id']), (int(detection['bbox'][0]), int(detection['bbox'][1])), cv2.FONT_HERSHEY_SIMPLEX, 0.005*min(detection['bbox'][2],detection['bbox'][3]), colors['0'], 5)
        # Get the body keypoints
        keypoints = detection['keypoints']

        # Loop over the keypoints and draw them
        for point in keypoints:
            x, y, _ = map(int, point)
            cv2.circle(copy, (x, y), 3, (0, 0, 255), -1)
    
    return copy

def gen():  
    sequence_kps = []
    people_label = {}
    agressive_people = []
    i = 0
    while True :
        # image = np.array(Image.open(frame))
        image = np.frombuffer(frame, dtype=np.uint8)

        image = image.reshape(480,640, 3)

        if not (image == images[-1]).all() :
            images.append(image)
            # -------------  YOLO-------------
            # ("added a frame to frame list")
            # ('current number of frames :',len(images))            
            results = det_model(image,classes=0,conf=0.4,iou=0.6,verbose=False)
            bbs=YOLO_preprocess(results[0].boxes.boxes.cpu().numpy())
           

        #  ------------------------ DeepSort -------------------------------
            tracks = tracker.update_tracks(bbs, frame=image) # bbs expected to be a list of detections, each in tuples of ( [left,top,w,h], confidence, detection_class )
            for track in tracks :
                (track.track_id)
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

            person_results = DS_preprocess(tracks)

            out, _ = inference_top_down_pose_model(pose_model,image,person_results,bbox_thr=0.4,format='xyxy')
            for j in range(len(out)):
                out[j]['bbox']=out[j]['bbox'].tolist()
                out[j]['keypoints']=out[j]['keypoints'].tolist()


            # ("\n------------------------------------\nresults of VitPose : ", [outs['track_id'] for outs in out],"\n")


            # Update VitPose outputs with track IDs
            for i,track in enumerate(tracks):
                if not track.is_confirmed():
                    continue
                out[i]['track_id'] = track.track_id
            
            sequence_kps.append(out)
            if len(sequence_kps) == 10:
                people_features = generate_features(sequence_kps,"dist")
                for people_key in people_features:
                    people_data = people_features[people_key]
                    people_data = np.asarray(people_data)
                    # Change depending on model we use, if CNN : shape[1]/2,2
                    people_data = people_data.reshape(1,people_data.shape[0],int(people_data.shape[1]/2),2)
                    prediction = classifier.predict(people_data)
                    label = np.argmax(prediction)
                    people_label[str(people_key)] = label
                    if label == 1 and people_key not in agressive_people:
                        agressive_people.append(people_key)
                        for vitpose_result in out : 
                            if vitpose_result['track_id'] == people_key:
                                bboxes = vitpose_result['bbox']
                                x= (bboxes[0]+bboxes[3])/2
                                y = (bboxes[1] + bboxes[2])/2
                        data = {
                            "EventId" : event_id,
                            "AlarmType" : "Conflict",
                            "Description" : "Conflict is happening",
                            "SensorId" : "Camera1",
                            "Priority" : 1,
                            "VirtualInterceptCoordinates" : "Conflict on position ("+str(x) + ','+str(y)+')',
                            "XPos" : None,
                            "YPos" : None
                        }
                        event_id+=1
                        m=json.dumps(data)
                        print(m)
                        p.poll(1)
                        p.produce('hildasafe01', m.encode('utf-8'),callback=receipt)
                        p.flush()
                        time.sleep(3)
                    else:
                        if people_key in agressive_people and label !=1 :
                            agressive_people.remove(people_key)
                sequence_kps = []
            image_with_output = draw_data_on_image(image,out,people_label)
            _, frame_with_output = cv2.imencode('.jpg', image_with_output)

            yield (b'--frame\r\nContent-Type: image/jpeg\r\n\r\n' + frame_with_output.tobytes() + b'\r\n')




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


if __name__ == "__main__":
    load_models()
    port = int(os.environ.get('PORT', 5000))
    app.run(debug=False, host='0.0.0.0', port=port)
    
