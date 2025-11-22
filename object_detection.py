import numpy as np
import cv2
import carla

class Object_Detection:
    def __init__(self):
        self.image = None
        print("Object_Detection class initialized")

    def callback(self, image: carla.Image):
        """ Runs everytime a new image from the sensor comes in """

        array = np.frombuffer(image.raw_data, dtype=np.uint8)   # Takes in the data as 1d buffer
        array = array.reshape((image.height, image.width, 4))   # Reshapes it to 4d

        frame = array[:, :, :3]  # Keep only the colour
        self.image = frame

        cv2.imshow("Camera Test", self.image)
        cv2.waitKey(1)


