import numpy as np
import cv2
from ultralytics import YOLO
from pymongo import MongoClient
import torch
from collections import OrderedDict
from threading import Thread, Lock

# CentroidTracker class
class CentroidTracker:
    def __init__(self):
        self.next_object_id = 0
        self.objects = OrderedDict()
        self.rectangles = OrderedDict()

    def register(self, centroid):
        self.objects[self.next_object_id] = centroid
        self.rectangles[self.next_object_id] = None
        self.next_object_id += 1

    def deregister(self, object_id):
        del self.objects[object_id]
        del self.rectangles[object_id]

    def update(self, detections):
        if len(detections) == 0:
            # If no detections, deregister all existing objects
            for object_id in list(self.objects.keys()):
                self.deregister(object_id)
            return self.objects

        centroids = np.zeros((len(detections), 2), dtype="int")

        for (i, (xmin, ymin, xmax, ymax)) in enumerate(detections):
            cX = int((xmin + xmax) / 2.0)
            cY = int((ymin + ymax) / 2.0)
            centroids[i] = (cX, cY)

        if len(self.objects) == 0:
            for i in range(len(centroids)):
                self.register(centroids[i])
        else:
            object_ids = list(self.objects.keys())
            object_centroids = list(self.objects.values())

            # Compute distance between each pair of object centroids
            D = np.linalg.norm(np.array(object_centroids) - centroids[:, np.newaxis], axis=2)

            # Perform the matching
            rows = D.min(axis=1).argsort()
            cols = D.argmin(axis=1)[rows]

            used_rows = set()
            used_cols = set()

            for (row, col) in zip(rows, cols):
                if row in used_rows or col in used_cols:
                    continue
                object_id = object_ids[row]
                self.objects[object_id] = centroids[col]
                self.rectangles[object_id] = detections[col]
                used_rows.add(row)
                used_cols.add(col)

            # Register new centroids that were not matched
            for i in range(len(centroids)):
                if i not in used_cols:
                    self.register(centroids[i])

            # Deregister objects that were not matched
            for object_id in list(self.objects.keys()):
                if object_id not in used_rows:
                    self.deregister(object_id)

        return self.objects

# MongoDB setup
hostname = '200.10.3.226'
port = 27017
username = 'fariz'
password = 'paris327'
database_name = 'iotmaker'

uri = f"mongodb://{username}:{password}@{hostname}:{port}/{database_name}?authSource=admin"
client = MongoClient(uri)

try:
    db = client[database_name]
    print("Koneksi berhasil!")
except Exception as e:
    print(f"Koneksi gagal: {e}")

collection = db["doors"]
updateperson = db["airconditioners"]

mutex = Lock()

def new1():
    with mutex:
        collection.update_one({"idlamp": 1}, {"$set": {"switch1": "on"}})

def new2():
    with mutex:
        collection.update_one({"idlamp": 1}, {"$set": {"switch1": "off"}})

def statorang():
    with mutex:
        updateperson.update_one({"ruangan": 1308}, {"$set": {"person": "kosong"}})

def addorang():
    with mutex:
        updateperson.update_one({"ruangan": 1308}, {"$set": {"person": "ada"}})

def itungorang(detected_persons):
    with mutex:
        updateperson.update_one({"ruangan": 1308}, {"$set": {"countingpeople": detected_persons}})

if __name__ == '__main__':
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    print('device:', device)
    model = YOLO("yolov8x-pose.pt").to(device)
    
    cap = cv2.VideoCapture("rtsp://admin:EIRGYD@200.10.3.94:5547", cv2.CAP_FFMPEG)
    tracker = CentroidTracker()

    while cap.isOpened():
        status, frame = cap.read()
        if not status:
            continue

        results = model(frame, stream=True, verbose=False, conf=0.5)

        detections = []
        for res in results:
            filtered_indices = np.where(res.boxes.conf.cpu().numpy() > 0.5)[0]
            boxes = res.boxes.xyxy.cpu().numpy()[filtered_indices].astype(int)
            for xmin, ymin, xmax, ymax in boxes:
                detections.append((xmin, ymin, xmax, ymax))

        # Update tracker with the latest detections
        objects = tracker.update(detections)
        detected_persons = len(objects)

        # Draw bounding boxes and IDs
        for object_id, centroid in objects.items():
            if object_id in tracker.rectangles and tracker.rectangles[object_id] is not None:
                # If the rectangle is not None, draw the bounding box
                (xmin, ymin, xmax, ymax) = tracker.rectangles[object_id]
                cv2.putText(frame, f"ID: {object_id}", (centroid[0], centroid[1]), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)
                cv2.rectangle(frame, (xmin, ymin), (xmax, ymax), (0, 255, 0), 2)

        print("Jumlah orang terdeteksi:", detected_persons)
        itungorang(detected_persons)
        
        cv2.imshow("frame", frame)

        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cap.release()