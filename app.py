import os
import cv2
import numpy as np
import base64
from flask import Flask, render_template_string, request, redirect, flash
import roboflow
import torch
from collections import Counter

app = Flask(__name__)
app.secret_key = 'your_secret_key'  # Replace with a secure secret key

#########################################
# 1. Initialize the Models
#########################################

# --- Roboflow Box Detection Model ---
API_KEY = "wLjPoPYaLmrqCIOFA0RH"            # Replace with your actual API key
PROJECT_ID = "base-model-box-r4suo-8lkk1-6dbqh"      # Replace with your Roboflow project ID
VERSION_NUMBER = "2"  # Replace with your trained model version number

rf = roboflow.Roboflow(api_key=API_KEY)
workspace = rf.workspace()
project = workspace.project(PROJECT_ID)
version = project.version(VERSION_NUMBER)
box_model = version.model  # This model is trained for detecting boxes

# --- YOLOv5 Pretrained Model for Persons & Cars ---
# Using Ultralytics YOLOv5s (pretrained) from Torch Hub
yolov5_model = torch.hub.load('ultralytics/yolov5', 'yolov5s', pretrained=True)
# We'll filter YOLO detections to only include persons and cars.
YOLO_FILTER_CLASSES = {"person", "car"}

#########################################
# 2. Helper Functions
#########################################

def compute_iou(boxA, boxB):
    xA = max(boxA[0], boxB[0])
    yA = max(boxA[1], boxB[1])
    xB = min(boxA[2], boxB[2])
    yB = min(boxA[3], boxB[3])
    interWidth = max(0, xB - xA)
    interHeight = max(0, yB - yA)
    interArea = interWidth * interHeight
    boxAArea = (boxA[2] - boxA[0]) * (boxA[3] - boxA[1])
    boxBArea = (boxB[2] - boxB[0]) * (boxB[3] - boxB[1])
    if boxAArea + boxBArea - interArea == 0:
        return 0
    return interArea / float(boxAArea + boxBArea - interArea)

def custom_nms(preds, iou_threshold=0.3):
    preds = sorted(preds, key=lambda x: x["confidence"], reverse=True)
    filtered_preds = []
    for pred in preds:
        keep = True
        for kept in filtered_preds:
            if compute_iou(pred["box"], kept["box"]) > iou_threshold:
                keep = False
                break
        if keep:
            filtered_preds.append(pred)
    return filtered_preds

def process_image(image_path):
    """
    Process the uploaded image using both detection pipelines:
      (a) Box detection via Roboflow (with measurement using an ArUco marker).
      (b) YOLOv5 detection for persons and cars.
    Returns the annotated image and a list of detection info dictionaries.
    """
    image = cv2.imread(image_path)
    if image is None:
        return None, "Could not read the image."
    img_height, img_width = image.shape[:2]
    
    detection_info = []  # List to hold all detection results for display

    # --- (a) Roboflow Box Detection & Measurement ---
    results = box_model.predict(image_path, confidence=50, overlap=30).json()
    predictions = results.get("predictions", [])
    processed_preds = []
    for prediction in predictions:
        x, y, width, height = prediction["x"], prediction["y"], prediction["width"], prediction["height"]
        x1 = int(round(x - width / 2))
        y1 = int(round(y - height / 2))
        x2 = int(round(x + width / 2))
        y2 = int(round(y + height / 2))
        # Clamp coordinates to image dimensions
        x1 = max(0, min(x1, img_width - 1))
        y1 = max(0, min(y1, img_height - 1))
        x2 = max(0, min(x2, img_width - 1))
        y2 = max(0, min(y2, img_height - 1))
        processed_preds.append({
            "box": (x1, y1, x2, y2),
            "class": prediction["class"],
            "confidence": prediction["confidence"]
        })
    box_detections = custom_nms(processed_preds, iou_threshold=0.3)
    
    # Detect ArUco marker for measurement (only applicable for boxes)
    marker_real_width_cm = 10.0  # The marker is 10cm x 10cm
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    aruco_dict = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_6X6_250)
    aruco_params = cv2.aruco.DetectorParameters()
    corners, ids, _ = cv2.aruco.detectMarkers(gray, aruco_dict, parameters=aruco_params)
    if ids is not None and len(corners) > 0:
        marker_corners = corners[0].reshape((4, 2))
        cv2.aruco.drawDetectedMarkers(image, corners, ids)
        marker_width_pixels = np.linalg.norm(marker_corners[0] - marker_corners[1])
        marker_height_pixels = np.linalg.norm(marker_corners[1] - marker_corners[2])
        marker_pixel_size = (marker_width_pixels + marker_height_pixels) / 2.0
        conversion_factor = marker_real_width_cm / marker_pixel_size
    else:
        conversion_factor = None

    # Draw box detections and record measurement info (only for boxes)
    for pred in box_detections:
        x1, y1, x2, y2 = pred["box"]
        label = pred["class"]
        confidence = pred["confidence"]
        cv2.rectangle(image, (x1, y1), (x2, y2), (0, 255, 0), 2)
        if conversion_factor is not None:
            box_width_pixels = x2 - x1
            box_height_pixels = y2 - y1
            box_width_cm = box_width_pixels * conversion_factor
            box_height_cm = box_height_pixels * conversion_factor
            size_text = f"{box_width_cm:.1f}x{box_height_cm:.1f} cm"
            detection_info.append({
                "class": label,
                "confidence": f"{confidence:.2f}",
                "width_cm": f"{box_width_cm:.1f}",
                "height_cm": f"{box_height_cm:.1f}"
            })
        else:
            size_text = ""
            detection_info.append({
                "class": label,
                "confidence": f"{confidence:.2f}",
                "width_cm": "N/A",
                "height_cm": "N/A"
            })
        text = f"{label} ({confidence:.2f}) {size_text}"
        (text_width, text_height), baseline = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
        cv2.rectangle(image, (x1, y1 - text_height - baseline - 5), (x1 + text_width, y1 - 5), (0, 255, 0), -1)
        cv2.putText(image, text, (x1, y1 - 5 - baseline), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)
    
    # --- (b) YOLOv5 for Persons & Cars ---
    # Convert image to RGB for YOLO (it expects RGB)
    img_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    yolo_results = yolov5_model(img_rgb)
    df = yolo_results.pandas().xyxy[0]
    for _, row in df.iterrows():
        if row['name'] in YOLO_FILTER_CLASSES:
            xmin = int(row['xmin'])
            ymin = int(row['ymin'])
            xmax = int(row['xmax'])
            ymax = int(row['ymax'])
            conf = row['confidence']
            label = row['name']
            cv2.rectangle(image, (xmin, ymin), (xmax, ymax), (255, 0, 0), 2)
            text = f"{label} ({conf:.2f})"
            (text_width, text_height), baseline = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
            cv2.rectangle(image, (xmin, ymin - text_height - baseline - 5), (xmin + text_width, ymin - 5), (255, 0, 0), -1)
            cv2.putText(image, text, (xmin, ymin - 5 - baseline), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)
            detection_info.append({
                "class": label,
                "confidence": f"{conf:.2f}",
                "width_cm": "N/A",
                "height_cm": "N/A"
            })
    
    # --- Build Top Summary Text ---
    detection_counts = Counter(det["class"] for det in detection_info)
    if detection_counts:
        top_text = ", ".join(f"{cls}: {count}" for cls, count in detection_counts.items())
        (info_width, info_height), info_baseline = cv2.getTextSize(top_text, cv2.FONT_HERSHEY_SIMPLEX, 1, 2)
        cv2.rectangle(image, (5, 5), (5 + info_width, 5 + info_height + info_baseline), (0, 255, 0), -1)
        cv2.putText(image, top_text, (5, 5 + info_height), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 2)
    
    return image, detection_info

#########################################
# 3. Flask Routes
#########################################

@app.route('/', methods=['GET', 'POST'])
def index():
    image_data = None
    detection_info = None
    if request.method == 'POST':
        if 'file' not in request.files:
            flash('No file part')
            return redirect(request.url)
        file = request.files['file']
        if file.filename == '':
            flash('No selected file')
            return redirect(request.url)
        upload_path = "uploaded.jpg"
        file.save(upload_path)
        processed_image, detection_info = process_image(upload_path)
        if processed_image is None:
            flash("Error processing image.")
        else:
            retval, buffer = cv2.imencode('.jpg', processed_image)
            image_data = base64.b64encode(buffer).decode('utf-8')
        os.remove(upload_path)
    return render_template_string('''
    <!doctype html>
    <html>
      <head>
        <title>Multi-Detection & Measurement</title>
        <!-- Bootstrap CSS -->
        <link rel="stylesheet" href="https://stackpath.bootstrapcdn.com/bootstrap/4.5.2/css/bootstrap.min.css">
        <style>
          body {
            background-color: #f8f9fa;
            font-family: "Segoe UI", Tahoma, Geneva, Verdana, sans-serif;
          }
          .container {
            margin-top: 30px;
          }
          .header {
            text-align: center;
            margin-bottom: 30px;
          }
          .card {
            margin-bottom: 30px;
          }
          .result-img {
            width: 100%;
            border: 1px solid #ddd;
            padding: 5px;
          }
          .table-responsive {
            margin-top: 20px;
          }
          .footer {
            text-align: center;
            font-size: 0.9em;
            color: #777;
            margin-top: 30px;
          }
        </style>
      </head>
      <body>
        <div class="container">
          <h1 class="header">Multi-Detection & Measurement</h1>
          <!-- Upload Form -->
          <div class="card">
            <div class="card-body">
              <form method="post" enctype="multipart/form-data">
                <div class="form-group">
                  <label for="file">Choose an image to upload:</label>
                  <input type="file" class="form-control-file" name="file" accept="image/*" id="file">
                </div>
                <button type="submit" class="btn btn-primary">Upload</button>
              </form>
              {% with messages = get_flashed_messages() %}
                {% if messages %}
                  <div class="alert alert-danger mt-3">
                    <ul>
                      {% for message in messages %}
                        <li>{{ message }}</li>
                      {% endfor %}
                    </ul>
                  </div>
                {% endif %}
              {% endwith %}
            </div>
          </div>
          {% if image_data or detection_info %}
          <div class="row">
            <div class="col-md-8">
              <div class="card">
                <div class="card-header">
                  Processed Image
                </div>
                <div class="card-body">
                  <img src="data:image/jpeg;base64,{{ image_data }}" alt="Processed Image" class="result-img">
                </div>
              </div>
            </div>
            <div class="col-md-4">
              <div class="card">
                <div class="card-header">
                  Detection Results
                </div>
                <div class="card-body">
                  <p>Total Results: <strong>{{ detection_info|length }}</strong></p>
                  <div class="table-responsive">
                    <table class="table table-striped table-bordered">
                      <thead class="thead-dark">
                        <tr>
                          <th>#</th>
                          <th>Class</th>
                          <th>Confidence</th>
                          <th>Width (cm)</th>
                          <th>Height (cm)</th>
                        </tr>
                      </thead>
                      <tbody>
                        {% for det in detection_info %}
                        <tr>
                          <td>{{ loop.index }}</td>
                          <td>{{ det.class }}</td>
                          <td>{{ det.confidence }}</td>
                          <td>{{ det.width_cm }}</td>
                          <td>{{ det.height_cm }}</td>
                        </tr>
                        {% endfor %}
                      </tbody>
                    </table>
                  </div>
                </div>
              </div>
            </div>
          </div>
          {% endif %}
          <div class="footer">
            <p>&copy; 2023 Multi-Detection App. All rights reserved.</p>
          </div>
        </div>
        <!-- Bootstrap JS and dependencies -->
        <script src="https://code.jquery.com/jquery-3.5.1.slim.min.js"></script>
        <script src="https://cdn.jsdelivr.net/npm/popper.js@1.16.1/dist/umd/popper.min.js"></script>
        <script src="https://stackpath.bootstrapcdn.com/bootstrap/4.5.2/js/bootstrap.min.js"></script>
      </body>
    </html>
    ''', image_data=image_data, detection_info=detection_info)

#########################################
# Run the App
#########################################

if __name__ == '__main__':
    app.run()
