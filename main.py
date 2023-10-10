# from test2 import main
# from face_saver import face_Saver
# import asyncio
# from threading import Thread
# from multiprocessing import Process,Queue,Lock
# from camera import Frames,cam1_frames


# if __name__=="__main__":
    
#     cam_thread=Thread(target=Frames)
#     main_thread=Thread(target=main)
#     face_thread=Thread(target=face_Saver)
   
#     cam_thread.start()
#     main_thread.start()
#     face_thread.start()


import cv2
import time
from pymongo import MongoClient
import numpy as np
from ultralytics import YOLO
import math 
import os
from rough import my
from pathlib import Path
from threading import Thread


# Open the video capture device (0 is typically the default camera)
cap = cv2.VideoCapture(0,cv2.CAP_DSHOW)

output_directory = "D:/deepface-master/deepface-master/api/recognized_faces"
os.makedirs(output_directory, exist_ok=True)

# Set the desired frame rate to display (normal frame rate)
desired_display_frame_rate = 500  # Adjust this to your desired display frame rate
display_frame_interval = int(1000 / desired_display_frame_rate)

# Set the frame rate to save to MongoDB (1 frame per second)
desired_save_frame_rate = 1  # 1 frame per second
save_frame_interval = int(1000 / desired_save_frame_rate)

# MongoDB setup
client = MongoClient("mongodb://localhost:27017")
db = client["cam_db"]
collection = db["cam_collection"]
collection_2 = db["face_attributes"]

folder_dir = Path("D:/deepface-master/deepface-master/api/recognized_faces").glob("*.jpg")

model = YOLO("yolo-Weights/yolov8n.pt")

classNames = ["person", "bicycle", "car", "motorbike", "aeroplane", "bus", "train", "truck", "boat",
              "traffic light", "fire hydrant", "stop sign", "parking meter", "bench", "bird", "cat",
              "dog", "horse", "sheep", "cow", "elephant", "bear", "zebra", "giraffe", "backpack", "umbrella",
              "handbag", "tie", "suitcase", "frisbee", "skis", "snowboard", "sports ball", "kite", "baseball bat",
              "baseball glove", "skateboard", "surfboard", "tennis racket", "bottle", "wine glass", "cup",
              "fork", "knife", "spoon", "bowl", "banana", "apple", "sandwich", "orange", "broccoli",
              "carrot", "hot dog", "pizza", "donut", "cake", "chair", "sofa", "pottedplant", "bed",
              "diningtable", "toilet", "tvmonitor", "laptop", "mouse", "remote", "keyboard", "cell phone",
              "microwave", "oven", "toaster", "sink", "refrigerator", "book", "clock", "vase", "scissors",
              "teddy bear", "hair drier", "toothbrush"
              ]

processed_images = set()
timer_interval = 1.0

def detector():   
    while True:
        folder_dir = Path("D:/deepface-master/deepface-master/api/recognized_faces").glob("*.jpg")
        # start_time = time.time()
        for image_path in folder_dir:
            image_path = str(image_path)  # Convert Path to string
            if image_path not in processed_images:
                start_time = time.time()
                data=my(image_path)
                # print(image_path)
                processed_images.add(image_path)
                collection_2.insert_one(data)
                elapsed_time = time.time() - start_time
            if elapsed_time < timer_interval:
                time.sleep(timer_interval - elapsed_time)


attributes_thread=Thread(target=detector)
attributes_thread.start()

frame_counter = 0
person_count=0
counter=0
start_time = time.time()
while True:
  
        ret, frame = cap.read()

        if not ret:
            break
        
        results = model(frame, stream=True)
     


        # Reset person_count for each frame
        person_count = 0

        for r in results:
            boxes = r.boxes
        
            for box in boxes:
                # bounding box
                x1, y1, x2, y2 = box.xyxy[0]
                x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2) # convert to int values
                # confidence
                confidence = math.ceil((box.conf[0]*100))/100
                # print("Confidence --->",confidence)
                # print(box.path)

                # class name
                cls = int(box.cls[0])
                # print("Class name -->", classNames[cls])
            
                # object details
                org = [x1, y1]
                font = cv2.FONT_HERSHEY_SIMPLEX
                fontScale = 1
                color = (255, 0, 0)
                thickness = 2

                if cls == 0:
                    # put box in cam
                    person_count += 1
                    cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 0, 255), 3)

                    cv2.putText(frame,classNames[cls], org, font, fontScale, color, thickness)

            # Save 1 frame per second to MongoDB  
            
            elapsed_time = time.time() - start_time
            
            if elapsed_time >= 1:
                collection.insert_one({"Person_count": person_count, "timestamp": time.asctime()})
                start_time = time.time()
                image = r.orig_img 
                
                # print(r.path)
                counter += 1
                unique_filename = f"unknown_face_{counter}{r.path}"
                output_path = os.path.join(output_directory, unique_filename)
                cv2.imwrite(output_path, image)
            
                
                # person_count = 0


            # Display frames at the normal frame rate
            if frame_counter % display_frame_interval == 0:
                cv2.imshow('Normal Frame Rate', frame)
            
            frame_counter += 1

   

        # Exit the loop if the 'q' key is pressed
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

# Release video object and close windows
cap.release()
cv2.destroyAllWindows()