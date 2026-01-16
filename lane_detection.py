import numpy as np
import cv2
import carla

from ultralytics import YOLO

class Lane_Detection:
    def __init__(self):
        self.image = None
        #self.model = YOLO("yolov8n.pt")
        self.confidence_threshold = 0.3
        #self.labels = ["car", "truck", "bus", "motorcycle", "bicycle", "person", "traffic light", "stop sign"]
        self.window = cv2.namedWindow("Lane_Detector", cv2.WINDOW_NORMAL)

        cv2.resizeWindow("Lane_Detector", 300, 300)
        print("Lane_Detection class initialized")

    def callback(self, image: carla.Image):
        """ Runs everytime a new image from the sensor comes in """

        array = np.frombuffer(image.raw_data, dtype=np.uint8)   # Takes in the data as 1d buffer
        array = array.reshape((image.height, image.width, 4))   # Reshapes it to 4d

        frame = array[:, :, :3].copy()  # Keep only the colour, also it was readonly
        self.image = frame


    def process_image(self):
        if self.image is None:
            return

        h, w = self.image.shape[:2]
        crop = self.image[h//2:h, 0:w]
        gray = cv2.cvtColor(crop, cv2.COLOR_RGB2GRAY)
        blur = cv2.GaussianBlur(gray, (5,5), 0)
        #_, binary = cv2.threshold(blur, 225, 255, cv2.THRESH_BINARY)

        cv2.imshow("Lane_Detector", blur)
        cv2.waitKey(1)
        
        


