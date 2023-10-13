# import cv2

# # Open the video capture device (0 is typically the default camera)
# cap = cv2.VideoCapture(0)

# # Set the frame rate to 1 frame per second
# desired_frame_rate = 1

# # Get the current frame rate
# current_frame_rate = int(cap.get(cv2.CAP_PROP_FPS))

# # Create a VideoWriter object to save the reduced frame rate video
# fourcc = cv2.VideoWriter_fourcc(*'XVID')  # Codec for saving video (may vary)
# output = cv2.VideoWriter('output_video.avi', fourcc, desired_frame_rate, (640, 480))  # Adjust resolution as needed

# frame_counter = 0

# while True:
#     ret, frame = cap.read()

#     if not ret:
#         break

#     # Increment the frame counter
#     frame_counter += 1

#     # Display the frame counter on the frame
#     cv2.putText(frame, f"Frame: {frame_counter}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

#     # Display the frame
#     cv2.imshow('Frame with Counter', frame)

#     # Write the frame to the output video
#     output.write(frame)

#     # Wait for a second to achieve 1 frame per second
#     cv2.waitKey(1000 // desired_frame_rate)

#     # Exit the loop if the 'q' key is pressed
#     if cv2.waitKey(1) & 0xFF == ord('q'):
#         break

# # Release video objects and close windows
# cap.release()
# output.release()
# cv2.destroyAllWindows()

# import cv2
# import time
# from pymongo import MongoClient
# import numpy as np
# from ultralytics import YOLO
# import math
# import os
# from pathlib import Path
# from threading import Thread
# from deepface import DeepFace
# from sort import Sort
# import requests

# # Initialize the video capture device (0 is typically the default camera)
# cap = cv2.VideoCapture("http://192.168.137.125:8080/video")

# # Create a directory to store recognized faces
# output_directory = "D:/deepface-master/deepface-master/api/recognized_faces"
# os.makedirs(output_directory, exist_ok=True)

# # Set the desired frame rate to display (normal frame rate)
# desired_display_frame_rate = 30  # Adjust this to your desired display frame rate
# display_frame_interval = int(1000 / desired_display_frame_rate)

# # Set the frame rate to save to MongoDB (1 frame per second)
# desired_save_frame_rate = 1  # 1 frame per second
# save_frame_interval = int(1000 / desired_save_frame_rate)

# # MongoDB setup
# client = MongoClient("mongodb://localhost:27017")
# db = client["cam_db"]
# collection = db["cam_collection"]
# collection_2 = db["face_attributes"]

# # Define the YOLO model for object detection
# model = YOLO("yolo-Weights/yolov8n.pt")

# # Class names for YOLO detection
# classNames = [
#     "person", "bicycle", "car", "motorbike", "aeroplane", "bus", "train", "truck", "boat",
#     # ... add other class names as needed
# ]

# # Set up variables for face recognition
# processed_images = set()
# face_attributes={}
# # Function to detect faces in images and store face attributes in MongoDB
# def detect_faces(image_path):
    
#     retrive=DeepFace.analyze(image_path,actions=["emotion", "age", "gender", "race"], enforce_detection=False)
#     result=retrive[0]
#     age = result["age"]
#     dominant_gender = result["dominant_gender"]
#     dominant_emotion = result["dominant_emotion"]
#     face_attributes.update({"Age": age, "Dominant Gender": dominant_gender, "Dominant Emotion": dominant_emotion})
#     print(f"Age: {age}, Dominant Gender: {dominant_gender}, Dominant Emotion: {dominant_emotion}")
#     collection_2.insert_one(face_attributes)

# # Function to process images for object detection and face recognition
# def process_images():
#     while True:
#         folder_dir = Path(output_directory).glob("*.jpg")
#         for image_path in folder_dir:
#             image_path = str(image_path)
#             if image_path not in processed_images:
#                 detect_faces(image_path)
#                 processed_images.add(image_path)

# # Start a thread to process images for face recognition
# attributes_thread = Thread(target=process_images)
# attributes_thread.start()

# # Initialize the SORT tracker
# tracker = Sort()

# frame_counter = 0
# person_count = 0
# counter = 0
# start_time = time.time()
# bounding_boxes = []

# while True:
#     try:
#         ret, frame = cap.read()

#         if not ret:
#             break

#         results = model(frame, stream=True)
#         all_bounding_boxes = []

#         # Reset person_count for each frame
#         person_count = 0

#         for r in results:
#             boxes = r.boxes

#             for box in boxes:
#                 # Bounding box coordinates
#                 x1, y1, x2, y2 = box.xyxy[0]
#                 x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)

#                 # Confidence of detection
#                 confidence = math.ceil((box.conf[0] * 100)) / 100

#                 # Class name of detected object
#                 cls = int(box.cls[0])

#                 # Draw bounding boxes for people (class index 0)
#                 if cls == 0:
#                     # Record the bounding box
#                     x1, y1, x2, y2 = map(int, box.xyxy[0])
#                     bounding_boxes.append([x1, y1, x2 - x1, y2 - y1])
#                     all_bounding_boxes.extend(bounding_boxes)
#                     person_count += 1

#             # Update object tracking using SORT tracker
#             if all_bounding_boxes:
#                 trackers = tracker.update(np.array(all_bounding_boxes))
#                 for d in trackers:
#                     x, y, w, h, camera_idx, unique_id = map(int, d)
#                     cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
#                     cv2.putText(frame, f"ID: {unique_id}", (x, y - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

#             # Save 1 frame per second to MongoDB
#             elapsed_time = time.time() - start_time
#             if elapsed_time >= 1:
#                 collection.insert_one({"Person_count": person_count, "timestamp": time.asctime()})
#                 start_time = time.time()
#                 image = r.orig_img
#                 counter += 1
#                 unique_filename = f"unknown_face_{counter}{r.path}"
#                 output_path = os.path.join(output_directory, unique_filename)
#                 cv2.imwrite(output_path, image)

#             # Display frames at the normal frame rate
#             if frame_counter % display_frame_interval == 0:
#                 cv2.imshow('Normal Frame Rate', frame)

#             frame_counter += 1

#     except Exception as e:
#         print(f"Error {e}")

        
#     # Exit the loop if the 'q' key is pressed
#     if cv2.waitKey(1) & 0xFF == ord('q'):
#         break

# # Release video object and close windows
# cap.release()
# cv2.destroyAllWindows()


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
from deepface import DeepFace
import face_recognition



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
collection_3=db["Value"]

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
recog_faces=set()
timer_interval = 1.0
known_face_encoding = []

def compare():
    while True:
        try:
            folder_dir = Path("D:/deepface-master/deepface-master/api/recognized_faces").glob("*.jpg")
            # start_time = time.time()
            for image_path in folder_dir:
                image_path = str(image_path)
                if image_path not in recog_faces:
                    for face in folder_dir:
                        face=str(face)

                        print(f"{face} comparing {image_path}")
                        
                        match=DeepFace.verify(image_path,face,model_name="Facenet",detector_backend= "mtcnn",distance_metric= "euclidean")
                        if match["verified"]==True: 
                          match_value = "True" 
                        else:
                            match_value = "False"
                        recog_faces.add(image_path)
                        collection_3.insert_one({"match": match_value})
        except Exception as e:
            print(f"Verify Error {e}")



def detector():
  
    while True:
        try:
            folder_dir = Path("D:/deepface-master/deepface-master/api/recognized_faces").glob("*.jpg")
            # start_time = time.time()
            for image_path in folder_dir:
                image_path = str(image_path)  # Convert Path to string
                
                if image_path not in processed_images:
                    start_time = time.time()
                    ret=DeepFace.analyze(image_path, actions=["emotion", "age", "gender"],enforce_detection=False)
                    # print(image_path)


                    processed_images.add(image_path)

                    result=ret[0]
                    age = result["age"]
                    dominant_gender = result["dominant_gender"]
                    dominant_emotion = result["dominant_emotion"]
                    
                    
                    collection_2.insert_one({"Age": age, "Dominant Gender": dominant_gender, "Dominant Emotion": dominant_emotion,"time":time.asctime()})

                    elapsed_time = time.time() - start_time
                if elapsed_time < timer_interval:
                    time.sleep(timer_interval - elapsed_time)


        except Exception as e:
            print(f'Error {e}')

attributes_thread=Thread(target=detector)
attributes_thread.start()

compare_thread=Thread(target=compare)
compare_thread.start()

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
            

            # Display frames at the normal frame rate
            if frame_counter % display_frame_interval == 0:
                cv2.imshow('Normal Frame Rate', frame)
            
            frame_counter += 1

   

        # Exit the loop if the 'q' key is pressed
        if cv2.waitKey(1) & 0xFF == ord('q'):
            exit()
        

# Release video object and close windows
cap.release()
cv2.destroyAllWindows()


