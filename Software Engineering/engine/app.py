from flask import Flask, request,Response, jsonify, render_template
import os
import io
import time
import cv2
import numpy as np
import PIL.Image as Image
import matplotlib.pyplot as plt


app = Flask(__name__)

images = [None]

def gen():
    while True :
        image = np.array(Image.open(io.BytesIO(frame))) 
        # yield (b'--frame\r\nContent-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

        if not (image == images[-1]).all() :
            images.append(image)
            print("added a frame to frame list")
            print(len(images))





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
    port = int(os.environ.get('PORT', 5000))
    app.run(debug=True, host='0.0.0.0', port=port)