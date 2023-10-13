import cv2
import time
from pymongo import MongoClient
import numpy as np
from ultralytics import YOLO
import math 
import os
# from rough import my
from pathlib import Path
from threading import Thread
from deepface import DeepFace
import face_recognition as fr
import uuid
import queue

# Open the video capture device (0 is typically the default camera)
class VStream:
    def __init__(self, src, window_name, window_size):
        self.capture = cv2.VideoCapture(src)
        # self.fps=int(self.capture.get(cv2.CAP_PROP_FPS))
        self.window_name = window_name
        self.window_size = window_size
        self.thread = Thread(target=self.update, args=())
        self.thread.daemon = True
        self.thread.start()
      

    def update(self):
        while True:
            _, self.frame = self.capture.read()

    def getFrame(self):
        return self.frame


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
collection_3=db["CustomerData"]

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
recog_faces={}
timer_interval = 1.0
known_face_encoding = []
locator_q=queue.Queue()

def locator():
    while True:
        try:
            folder_dir = Path("D:/deepface-master/deepface-master/api/recognized_faces").glob("*.jpg")
            for image_path in folder_dir:
                image_path_str = str(image_path)
                if image_path_str not in processed_images:
                    # Load the image using OpenCV
                    image = fr.load_image_file(image_path_str)
                    
                    # Find face locations in the image
                    face_locations = fr.face_locations(image)
                    
                    locator_q.put((image,face_locations))
        except Exception as e:
            print(f"Locator Error {e}")


encoder_q=queue.Queue()
def encoder():
    while True:
        try:
            image,face_locations=locator_q.get()
            # Find face encodings in the image
            face_encodings = fr.face_encodings(image, face_locations)
            encoder_q.put(face_encodings)
        except Exception as e:
            print(f"Encoder Error {e}")


#Compare function compares the face with recent frame and 
compare_q=queue.Queue()
def compare():
    while True:
        try:
            face_encodings=encoder_q.get()
            values=recog_faces.values()
            # Update the recognized faces dictionary with face encodings
            for face_encoding in face_encodings:
                
                match = fr.compare_faces(list(values),face_encoding)
    
                if any(match):
                    matching_keys = [key for key, is_match in zip(recog_faces.keys(), match) if is_match]
                    for matching_key in matching_keys:
                        compare_q.put(matching_keys)
                        print(f"Matched a known face with key: {matching_key}")
                else:
                    # Generate a unique UUID for each face
                    unique_id = str(uuid.uuid4())
                    # Add the face encoding to the recognized faces dictionary
                    recog_faces[unique_id] = face_encoding
                
        except Exception as e:
            print(f"Compare Error {e}")


#Detector function detectes different face attributes
def detector():
    while True:
        persons=compare_q.get()
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
                    
                    for person in persons:
                        collection_3.insert_one({"Person_id":person,"Age": age, "Dominant Gender": dominant_gender, "Dominant Emotion": dominant_emotion,"time":time.asctime()})

                        elapsed_time = time.time() - start_time
                        if elapsed_time < timer_interval:
                            time.sleep(timer_interval - elapsed_time)  
           
        except Exception as e:
            print(f'Error {e}')

attributes_thread=Thread(target=detector)
attributes_thread.start()

locator_thread=Thread(target=locator)
locator_thread.start()

compare_thread=Thread(target=compare)
compare_thread.start()

encoder_thread=Thread(target=encoder)
encoder_thread.start()

frame_counter = 0
person_count=0
counter=0
start_time = time.time()

window_sizes = {

    'Camera 2': (640, 480),
    'Camera 3': (640, 480)
}
phone_cam2 = VStream("http://192.168.137.125:8080/video", 'Camera 2', window_sizes['Camera 2'])
# phone_cam2 = VStream("http:/[2402:8100:795c:2915:0:d:2843:ea01]:8080/video", 'Camera 2', window_sizes['Camera 2'])
phone_cam3 = VStream(0, 'Camera 3', window_sizes['Camera 3'])

while True:
        try:
            phone1 = phone_cam2.getFrame()
            phone2= phone_cam3.getFrame()
            
            for frame, frame_name in [ (phone1, 'Camera 2'),(phone2, 'Camera 3')]:
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
                    # if frame_counter % display_frame_interval == 0:
                        if frame_name == 'Camera 2':
                            frame=cv2.resize(frame,window_sizes[frame_name])
                            cv2.imshow('Camera 2', frame)
                        elif frame_name == 'Camera 3':
                            frame=cv2.resize(frame,window_sizes[frame_name])
                            cv2.imshow('Camera 3', frame)
                    frame_counter += 1

   
        except Exception as e:
            print(f"Error {e}")
        # Exit the loop if the 'q' key is pressed
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
        

# Release video object and close windows
phone_cam2.capture.release()
phone_cam3.capture.release()
cv2.destroyAllWindows()


