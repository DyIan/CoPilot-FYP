import numpy as np
import cv2
import carla
import torch
from PIL import Image
from torchvision import transforms
from enet import ENet


class Lane_Detection:
    def __init__(self):
        self.image = None
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        self.model_path = "best_enet.pth"
        
        self.model = ENet(num_classes=3)
        self.model.load_state_dict(torch.load(self.model_path, map_location=self.device))
        self.model.to(self.device)
        self.model.eval()

        self.transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225]),
        ])

        self.window = cv2.namedWindow("Lane_Detector", cv2.WINDOW_NORMAL)

        cv2.resizeWindow("Lane_Detector", 600, 400)
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

        frame = self.image
        pil_image = Image.fromarray(frame)
        input_image = pil_image.resize((256,256))
        input_tensor = self.transform(input_image).unsqueeze(0).to(self.device)

        with torch.no_grad():
            output = self.model(input_tensor)
            prediction = torch.argmax(output, dim=1).squeeze(0).cpu().numpy()
        
        # Resize Prediction
        mask_resize = cv2.resize(prediction.astype(np.uint8), (frame.shape[1], frame.shape[0]), interpolation=cv2.INTER_NEAREST)

        # Creating the overlay
        mask = np.zeros_like(frame)


        mask[mask_resize == 1] = [0, 255, 0]    # Road is green
        mask[mask_resize == 2] = [0, 0, 255]    # Lane is red

        # Blend with original frame
        overlay = cv2.addWeighted(frame, 0.7, mask, 0.3, 0)

        cv2.imshow("Lane_Detector", overlay)
        cv2.waitKey(1)

        
        
        


