import os
import torch
from flask import Flask, request, render_template, redirect, url_for
import cv2
import numpy as np
from ultralytics import YOLO

# Load YOLOv8 model
model = YOLO('yolov8x.pt')  # Use yolov8x.pt for better accuracy (x means the largest model)

# Flask app
app = Flask(__name__)

# Ensure directories exist
UPLOAD_FOLDER = 'uploads'
RESULTS_FOLDER = os.path.join('static', 'results')

# Create directories if they don't exist
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(RESULTS_FOLDER, exist_ok=True)

# function to process image and detect objects
def detect_objects(image_path):
    img = cv2.imread(image_path)
    results = model(img)  # Perform detection using YOLOv8
    return results

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/upload', methods=['POST'])
def upload_file():
    if 'file' not in request.files:
        return redirect(url_for('home'))
    file = request.files['file']
    if file.filename == '':
        return redirect(url_for('home'))
    
    # Save the uploaded file
    image_path = os.path.join(UPLOAD_FOLDER, file.filename)
    file.save(image_path)
    
    # Detect objects in the image
    results = detect_objects(image_path)
    
    # Save the result image with detections
    results_img = np.squeeze(results[0].plot())  # Plot detections on the image
    result_image_path = os.path.join(RESULTS_FOLDER, file.filename)
    cv2.imwrite(result_image_path, results_img)
    
    return render_template('result.html', result_image=file.filename, boxes=results[0].boxes.xywh.tolist())

if __name__ == '__main__':
    app.run(debug=True)
