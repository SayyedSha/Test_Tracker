import cv2
from threading import Thread
from ultralytics import YOLO
import math
from pymongo import MongoClient
import time
import os
from pathlib import Path
from deepface import DeepFace
import queue
import face_recognition as fr
import uuid

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


output_directory = "D:/FFR/recognized_faces"
os.makedirs(output_directory, exist_ok=True)

#MongoDB connection
client = MongoClient("mongodb://localhost:27017")
db = client["cam_db"]
collection = db["person_count"]
collection_2 = db["face_attributes"]

#Making object for YOLO object-detection
model = YOLO("yolo-Weights/yolov8n.pt").cuda(device=0)


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


# camera_sources = [
#     {"source": "http://192.168.137.125:8080/video", "name": "Camera 2", "size": (640, 480)},
#     {"source": 0, "name": "Camera 3", "size": (640, 480)}
# ]


# vstreams = []

# # Create VStream instances for each camera source
# for camera_info in camera_sources:
#     source = camera_info["source"]
#     name = camera_info["name"]
#     size = camera_info["size"]
    
#     vstream = VStream(source, name, size)
#     vstreams.append(vstream)

# while True:
#     try:
#         frames = [vstream.getFrame() for vstream in vstreams]

#         for frame, frame_name in zip(frames, [info["name"] for info in camera_sources]):

#             results = model(frame, stream=True)

#             # Reset person_count for each frame
#             person_count = 0

#             for r in results:
#                 boxes = r.boxes
            
#                 for box in boxes:
#                     # bounding box
#                     x1, y1, x2, y2 = box.xyxy[0]
#                     x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2) # convert to int values
#                     # confidence
#                     confidence = math.ceil((box.conf[0]*100))/100
#                     # print("Confidence --->",confidence)
#                     # print(box.path)

#                     # class name
#                     cls = int(box.cls[0])
#                     # print("Class name -->", classNames[cls])
                
#                     # object details
#                     org = [x1, y1]
#                     font = cv2.FONT_HERSHEY_SIMPLEX
#                     fontScale = 1
#                     color = (255, 0, 0)
#                     thickness = 2

#                     if cls == 0:
#                         # put box in cam
#                         person_count += 1
#                         cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 0, 255), 3)

#                         cv2.putText(frame,classNames[cls], org, font, fontScale, color, thickness)

#             # Display frames with their respective names
#             frame = cv2.resize(frame, camera_sources[0]["size"])
#             cv2.imshow(frame_name, frame)

#     except Exception as E:
#         print(f"Error {E}")

#     # Handle keyboard input
#     if cv2.waitKey(1) & 0xFF == ord('q'):
#         break

# # Release video objects and close windows
# for vstream in vstreams:
#     vstream.capture.release()
# cv2.destroyAllWindows()

#Sources for video capture



processed_images = set()
timer_interval = 1.0   
known_face_encoding = []
recog_faces={}
locator_q=queue.Queue()
camera_q=queue.Queue()

def locator():
    while True:
        try:
            folder_dir = Path("D:/FFR/recognized_faces").glob("*.jpg")
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


def detector():
    # cameras= camera_q.get()
    persons=compare_q.get()
    while True:
       
        try:

            folder_dir = Path("D:/FFR/recognized_faces").glob("*.jpg")
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
                    
                    # for person,camera in zip(persons,cameras):
                    for person in persons:
                        collection_2.insert_one({"Camera":image_path,"Person":person,"Age": age, "Dominant Gender": dominant_gender, "Dominant Emotion": dominant_emotion,"time":time.asctime()})
            
                    elapsed_time = time.time() - start_time
                    if elapsed_time < timer_interval:
                        time.sleep(timer_interval - elapsed_time)  
           
        except Exception as e:
            print(f'Error {e}')

def main():
    camera_sources = [
        {"source": "http://192.168.137.125:8080/video", "name": "Camera 1", "size": (640, 480)},
        # {"source": "http://192.168.137.5:8080/video", "name": "Camera 2", "size": (640, 480)},
        {"source": 0, "name": "Camera 3", "size": (640, 480)}
    ]

    vstreams = []

    # Create VStream instances for each camera source
    for camera_info in camera_sources:
        source = camera_info["source"]
        name = camera_info["name"]
        size = camera_info["size"]
        
        vstream = VStream(source, name, size)
        vstreams.append(vstream)

    # Initialize person count for each camera
    person_counts = [0] * len(camera_sources)
    start_time = time.time()
    counter=0
    


    # Running main loop
    while True:
        try:
            frames = [vstream.getFrame() for vstream in vstreams]
            # camera_q.put(frame)
            for i, (frame, frame_name) in enumerate(zip(frames, [info["name"] for info in camera_sources])):

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
                        

                        # class name
                        cls = int(box.cls[0])
                        
                    
                        # object details
                        org = [x1, y1]
                        font = cv2.FONT_HERSHEY_SIMPLEX
                        fontScale = 1
                        color = (255, 0, 0)
                        thickness = 2

                        if cls == 0:
                            # Count persons
                            person_count += 1

                            # Put box in cam
                            cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 0, 255), 3)
                            cv2.putText(frame, classNames[cls], org, font, fontScale, color, thickness)

                # Update person count for the camera
                person_counts[i] = person_count

                # Display frames with person count
                text = f"{frame_name}: Persons Detected: {person_count}"
                frame = cv2.resize(frame, camera_sources[i]["size"])
                cv2.putText(frame, text, (10, 30), font, 1, color, thickness)
                cv2.imshow(frame_name, frame)


                elapsed_time = time.time() - start_time
                #Storing data one frame per second      
                if elapsed_time >= 1:
                    # camera_q.put([frame_name])
                    collection.insert_one({"Camera": frame_name, "Person_count": person_count, "timestamp": time.asctime()})
                    start_time = time.time()

                    #Storing frame images in a folder
                    image = r.orig_img
                    counter += 1
                    unique_filename = f"{frame_name}_{counter}.jpg"
                    output_path = os.path.join(output_directory, unique_filename)
                    cv2.imwrite(output_path, image)

        except Exception as E:
            print(f"Error {E}")

        # Handle keyboard input
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # Release video objects and close windows
    for vstream in vstreams:
        vstream.capture.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    detector_Thread = Thread(target=detector)
    detector_Thread.start()
    main_Thread = Thread(target=main)
    main_Thread.start()
    locator_thread=Thread(target=locator)
    locator_thread.start()
    compare_thread=Thread(target=compare)
    compare_thread.start()
    encoder_thread=Thread(target=encoder)
    encoder_thread.start()