import cv2
import numpy as np


LANE_CLASS = 2
mask = cv2.imread("mask_test4.png", cv2.IMREAD_GRAYSCALE)
h, w = mask.shape

# Extract the lane info
lane_mask = (mask == LANE_CLASS).astype(np.uint8)
roi = lane_mask[int(0.6 * h):h, :]

# 
center_x = roi.shape[1] // 2
lookahead_y = 0.3


ys = []
left_pts = []
right_pts = []

visualise = np.zeros((roi.shape[0], roi.shape[1], 3), dtype=np.uint8)

for y in range(roi.shape[0]):
    xs = np.where(roi[y] > 0)[0]
    if len(xs) == 0:
        continue

    left_xs = xs[xs < center_x]
    right_xs = xs[xs > center_x]

    if len(left_xs) == 0 or len(right_xs) == 0:
        continue

    left = left_xs[-1]
    right = right_xs[0]

    left_pts.append(left)
    right_pts.append(right)
    ys.append(y / roi.shape[0])

if len(ys) > 10:
    left_fit = np.polyfit(ys, left_pts, 2)
    right_fit = np.polyfit(ys, right_pts, 2)
    center_fit = (left_fit + right_fit) / 2
else:
    left_fit = right_fit = center_fit = None


target_x = np.polyval(center_fit, lookahead_y)
steering_error = target_x - center_x

if center_fit is not None:
    for y in range(roi.shape[0]):
        y_norm = y / roi.shape[0]

        left_x = int(np.polyval(left_fit, y_norm))
        right_x = int(np.polyval(right_fit, y_norm))
        center_x_fit = int(np.polyval(center_fit, y_norm))

        # Draw Lanes
        if 0 <= left_x < roi.shape[1]:
            visualise[y, left_x] = (255, 0, 0)
        if 0 <= right_x < roi.shape[1]:
            visualise[y, right_x] = (0, 255, 0)
        if 0 <= center_x_fit < roi.shape[1]:
            visualise[y, center_x_fit] = (0, 0, 255)


# Draw Lookahead
y_px = int(lookahead_y * roi.shape[0])
cv2.circle(visualise, (int(target_x), y_px), 5, (0, 255, 255), -1)
print(f"Steering Error: {steering_error}")
        

cv2.line(visualise, (center_x, 0), (center_x, roi.shape[0]), (255, 255, 255), 1)
cv2.imshow("Lane", visualise)
cv2.imshow("Lane2", roi * 255)
cv2.waitKey(0)
