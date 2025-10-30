🧠 Object Detection using OpenCV

This project performs real-time object detection using Python and OpenCV, powered by the SSD MobileNet deep learning model. It can detect and label multiple objects from images, videos, or live webcam feed.

🚀 Features

Detects multiple objects in real-time

Supports webcam, image, and video input

Uses pre-trained SSD MobileNet model for accurate detection

⚙️ Requirements

Install dependencies using:

pip install opencv-python

▶️ Run the Project

To detect objects using your webcam:

python objectDetection.py


or open and run the notebook:

objectDetection.ipynb

📸 Model Files

Ensure the following files are in the project folder:

frozen_inference_graph.pb

ssd_mobilenet_v3_large_coco_2020_01_14.pbtxt

labels.txt
