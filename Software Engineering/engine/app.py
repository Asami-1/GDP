from flask import Flask, request,Response, jsonify, render_template
import os
import io
import numpy as np
import PIL.Image as Image
import matplotlib.pyplot as plt
from ultralytics import YOLO
from custom_tracking import custom_tracking
import cv2

from mmpose.apis import (inference_top_down_pose_model, init_pose_model,
                         process_mmdet_results, vis_pose_result)

det_model = YOLO("yolov8n.pt")

pose_config = "./configs/ViTPose_huge_simple_coco_256x192.py"
pose_checkpoint = "./vitpose-h-simple.pth"
pose_model = None

app = Flask(__name__)


frames = [None]
images = [None]

def load_models():
    global det_model, pose_model
    pose_model = init_pose_model(pose_config,pose_checkpoint)


def process_yolo_results (yolo_results):

    results=yolo_results[0].boxes.boxes.to('cpu').numpy()

    person_results=[]
    for i in range(len(results)):
        person = {}
        person['bbox'] = results[i][range(5)]
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
    FirstDetected=False
    SecondDetection=False
    length=15
    list_lastouts=[[] for i in range(length)]
    while True :
        image = np.array(Image.open(io.BytesIO(frame)))

        if not (image == images[-1]).all() :
            images.append(image)
            print("added a frame to frame list")
            print('current number of frames :',len(images))            
            results = det_model(image)
            person_results= process_yolo_results(results)

            out, _ = inference_top_down_pose_model(pose_model,image,person_results,bbox_thr=0.5,format='xyxy')
            for j in range(len(out)):
                out[j]['bbox']=out[j]['bbox'].tolist()
                out[j]['keypoints']=out[j]['keypoints'].tolist()


            if not FirstDetected:
                detected_people=len(out)
                if detected_people>0:
                    FirstDetected=True
                    for i in range(detected_people):
                        out[i]['track_id']=i
                    next_id=detected_people
                    list_lastouts[0]=out

            else:
                # After first frame, compute tracking from previous frames         
                detected_people=len(out)
                if detected_people>0:
                    assigned_IDs=[]
                    assigned_bboxes=[]
                    notassigned_bboxes=list(range(len(out)))
                    for l in range(len(list_lastouts)):
                        if len(assigned_bboxes)!=detected_people:
                            lastout=list_lastouts[l]                    
                            people=len(lastout)
                            if people>0:
                                aux_out=out.copy()
                                bbox_relation=list(range(len(aux_out)))
                                aux_lastout=lastout.copy()
                                for i in reversed(range(len(aux_out))):
                                    if i in assigned_bboxes:
                                        del aux_out[i]
                                        del bbox_relation[i]
                                for i in reversed(range(len(aux_lastout))):
                                    if aux_lastout[i]['track_id'] in assigned_IDs:
                                        del aux_lastout[i]
                                aux_out,aux_lastout=custom_tracking(aux_out,aux_lastout,min_keypoints=3,use_oks=False,tracking_thr=0.3,use_one_euro=False,fps=None)
                                for i in range(len(aux_out)):
                                    if aux_out[i]['track_id'] !=-1:
                                        out[bbox_relation[i]]['track_id']=aux_out[i]['track_id']
                                        assigned_IDs.append(aux_out[i]['track_id'])
                                        assigned_bboxes.append(bbox_relation[i])
                                        notassigned_bboxes.remove(bbox_relation[i])

                    for i in range(len(notassigned_bboxes)):
                        out[notassigned_bboxes[i]]['track_id']=next_id
                        next_id+=1

                    # Update list of lastouts
                    for i in range(length):
                        if i==4:
                            list_lastouts[0]=out.copy()
                        else:
                            list_lastouts[4-i]=list_lastouts[3-i].copy()
            print("final results out :",out)
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