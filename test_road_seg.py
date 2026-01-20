import cv2
import torch
import numpy as np
from torchvision import transforms
from PIL import Image
from enet import ENet

# Device setup
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Load our model
model = ENet(num_classes=2)
model.load_state_dict(torch.load('best_enet.pth', map_location=device))
model.to(device)
model.eval()

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
    
    # Load Original image for comparison
    original = cv2.imread(image_path)
    original = cv2.resize(original, resize)

    # Create the mask
    mask = np.zeros_like(original)
    mask[pred == 1] = [0, 255, 0]

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

    # Resize prediction mask back to original frame size
    pred_resized = cv2.resize(pred.astype(np.uint8), (frame.shape[1], frame.shape[0]), interpolation=cv2.INTER_NEAREST)

    mask = np.zeros_like(frame)
    mask[pred_resized == 1] = [0, 255, 0]  # Green

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
    overlay_prediction("data/rgb/image.png", model, device)