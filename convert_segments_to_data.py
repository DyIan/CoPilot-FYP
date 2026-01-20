import numpy as np
import cv2
import os

# Load in images as unchanged
files = sorted([f for f in os.listdir(r"data\seg") if f.endswith(".png")])
print(f"Found {len(files)} segmentation files")




# Convert Colour To Classes
road_bgr = [128,64,128]
lane_bgr = [50,234,157]



for filename in files:
    seg_path = os.path.join(r"data\seg", filename)
    mask_path = os.path.join(r"data\mask", filename)

    seg_img = cv2.imread(seg_path, cv2.IMREAD_UNCHANGED)

    if seg_img is None:
        continue

    seg_img_rgb = seg_img[:, :, :3] # Keep bgr only no alpha

    # Create an empty mask
    mask = np.zeros(seg_img.shape[:2], dtype=np.uint8)

    # Set Mask pixels to 1 where class is road or lane
    road_mask = np.all(seg_img_rgb == road_bgr, axis=2)
    lane_mask = np.all(seg_img_rgb == lane_bgr, axis=2)
    
    # Assign Classes to each
    mask[road_mask] = 1
    mask[lane_mask] = 2

    cv2.imwrite(mask_path, mask)

