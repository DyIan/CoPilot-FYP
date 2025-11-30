import numpy as np
import cv2
import carla

from ultralytics import YOLO

class Object_Detection:
    def __init__(self):
        self.image = None
        self.model = YOLO("yolov8n.pt")
        self.confidence_threshold = 0.3
        self.labels = ["car", "truck", "bus", "motorcycle", "bicycle", "person", "traffic light", "stop sign"]
        self.window = cv2.namedWindow("Car_Detector", cv2.WINDOW_NORMAL)

        cv2.resizeWindow("Car_Detector", 300, 300)
        print("Object_Detection class initialized")

    def callback(self, image: carla.Image):
        """ Runs everytime a new image from the sensor comes in """

        array = np.frombuffer(image.raw_data, dtype=np.uint8)   # Takes in the data as 1d buffer
        array = array.reshape((image.height, image.width, 4))   # Reshapes it to 4d

        frame = array[:, :, :3].copy()  # Keep only the colour, also it was readonly
        self.image = frame


    def process_image(self):
        if self.image is None:
            return

        # Run yolo on the frame
        results = self.model(self.image, verbose=False)[0]

        # Draw a box around the image
        for box in results.boxes:
            cls = int(box.cls[0])
            label = results.names[cls]
            conf = float(box.conf[0])
            
            # Check if its confident on its prediction
            if conf < self.confidence_threshold:
                continue

            if label in self.labels:
                x1, y1, x2, y2 = box.xyxy[0].cpu().numpy().astype(int)

                # Draw the rectangle
                cv2.rectangle(self.image, (x1, y1), (x2, y2), (0,255,0), 2)

                # Put label on it
                cv2.putText(self.image, f"{label} {conf:.2f}", (x1, y1 - 5), cv2.FONT_HERSHEY_SCRIPT_SIMPLEX, 0.5, (0,255.0), 2)

        cv2.imshow("Car_Detector", self.image)
        cv2.waitKey(1)
        
        


