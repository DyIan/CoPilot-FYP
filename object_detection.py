import numpy as np
import cv2
import carla

class Object_Detection:
    def __init__(self):
        self.image = None
        print("Object_Detection class initialized")

    def callback(self, image: carla.Image):
        """ Runs everytime a new image from the sensor comes in """
        #print(f"Received image: {image.width}x{image.height}, frame: {image.frame}")
        # Must convert from CARLA BGRA to numpy RGB
        array = np.frombuffer(image.raw_data, dtype=np.uint8)
        array = array.reshape((image.height, image.width, 4))

        frame = array[:, :, :3]  # Keep only the colour
        #frame = frame[:, :, ::-1] # Reverse to RGB
        self.image = frame

        cv2.imshow("Camera Test", self.image)
        cv2.waitKey(1)


