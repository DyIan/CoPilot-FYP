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

centers = []
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
    center = int((left + right) / 2)

    visualise[y, left] = (255, 0, 0)
    visualise[y, right] = (0, 255, 0)
    visualise[y, center] = (0, 0, 255)

cv2.line(visualise, (center_x, 0), (center_x, roi.shape[0]), (255, 255, 255), 1)
cv2.imshow("Lane", visualise)
cv2.imshow("Lane2", roi * 255)
cv2.waitKey(0)
