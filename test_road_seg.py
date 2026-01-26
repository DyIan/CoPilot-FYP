import cv2
import torch
import numpy as np
from torchvision import transforms
from PIL import Image
from enet import ENet

# Device setup
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Load our model
model = ENet(num_classes=3)
model.load_state_dict(torch.load('best_enet.pth', map_location=device))
model.to(device)
model.eval()

ROAD_THRESH = 0.7
LANE_THRESH = 0.7

# Preprocess Image
def preprocess_image(image_path, resize=(256,256)):
    image = Image.open(image_path).convert('RGB')
    image = image.resize(resize)

    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                             std=[0.229, 0.224, 0.225]),
    ])
    return transform(image).unsqueeze(0)

def overlay_prediction(image_path, model, device, resize=(256,256), alpha=0.5):
    input_tensor = preprocess_image(image_path, resize).to(device)

    # Run the model
    with torch.no_grad():
        output = model(input_tensor)
        pred = torch.argmax(output, dim=1).squeeze(0).cpu().numpy()
    
        
    mask = pred.astype(np.uint8)  # values: 0,1,2
    cv2.imwrite("mask_test4.png", mask)
    # Load Original image for comparison
    original = cv2.imread(image_path)
    original = cv2.resize(original, resize)

    # Create the mask
    mask = np.zeros_like(original)
    mask[pred == 1] = [0, 255, 0]
    mask[pred == 2] = [255, 0, 0]

    # Overlay
    overlay = cv2.addWeighted(original, 1- alpha, mask, alpha, 0)

    # Show 
    cv2.imshow("Road Segment", overlay)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


""" For Images

overlay_prediction("data/rgb/70201.png", model, device)

 """
def overlay_prediction_frame(frame, model, device, resize=(256,256), alpha=0.5):
    # Preprocess frame (NumPy array)
    image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    image = Image.fromarray(image)
    image = image.resize(resize)
    
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                             std=[0.229, 0.224, 0.225]),
    ])
    input_tensor = transform(image).unsqueeze(0).to(device)

    with torch.no_grad():
        output = model(input_tensor)
        pred = torch.argmax(output, dim=1).squeeze(0).cpu().numpy()
        probs = torch.softmax(output, dim=1)
    


    # Take probabilities
    conf = True
    if conf:
        road_prob = probs[0, 1].cpu().numpy()
        lane_prob = probs[0, 2].cpu().numpy()

        road_mask = road_prob > ROAD_THRESH
        lane_mask = lane_prob > LANE_THRESH

        road_resized = cv2.resize(road_mask.astype(np.uint8), (frame.shape[1], frame.shape[0]), interpolation=cv2.INTER_NEAREST)
        lane_resized = cv2.resize(lane_mask.astype(np.uint8), (frame.shape[1], frame.shape[0]), interpolation=cv2.INTER_NEAREST)

        # Create overlay
        overlay = frame.copy()

        overlay[road_resized == 1] = [0, 255, 0]  # Green
        overlay[lane_resized == 1] = [0, 0, 255]  # Red
        overlay = cv2.addWeighted(frame, 1 - alpha, overlay, alpha, 0)
        return overlay

    # Resize prediction mask back to original frame size
    pred_resized = cv2.resize(pred.astype(np.uint8), (frame.shape[1], frame.shape[0]), interpolation=cv2.INTER_NEAREST)

    mask = np.zeros_like(frame)
    mask[pred_resized == 1] = [0, 255, 0]  # Green
    mask[pred_resized == 2] = [255, 0, 0]  # Red

    overlay = cv2.addWeighted(frame, 1 - alpha, mask, alpha, 0)
    return overlay

def process_vid():
    cap = cv2.VideoCapture('Test1.AVI')
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        overlayed = overlay_prediction_frame(frame, model, device)

        cv2.imshow("Road Segmentation Overlay", overlayed)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    #process_vid()
    overlay_prediction(r"30785.png", model, device)