import cv2
import numpy as np


LANE_CLASS = 2
mask = cv2.imread("mask_test4.png", cv2.IMREAD_GRAYSCALE)
h, w = mask.shape

# Extract the lane info
lane_mask = (mask == LANE_CLASS).astype(np.uint8)
roi = lane_mask[int(0.6 * h):h, :]
num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(
    roi, connectivity=8
)
solid_lane_mask = np.zeros_like(roi)
MIN_SOLID_VERTICAL_SPAN = int(0.3 * roi.shape[0])
MIN_SOLID_PIXEL_COUNT = 150

dashed_lane_mask = np.zeros_like(roi)
MIN_DASHED_VERTICAL_SPAN = int(0.1 * roi.shape[0])
MIN_DASHED_PIXEL_COUNT = 20



solid_components = []
dashed_components = []

for i in range(1, num_labels):  # skip background
    x, y, w, h, area = stats[i]

    if h >= MIN_SOLID_VERTICAL_SPAN and area >= MIN_SOLID_PIXEL_COUNT:
        solid_components.append(i)
    elif h >= MIN_DASHED_VERTICAL_SPAN and area >= MIN_DASHED_PIXEL_COUNT:
        dashed_components.append(i)

# Created a solid lines mask
for label in solid_components:
    solid_lane_mask[labels == label] = 1

# Create a dashed lines mask
for label in dashed_components:
    dashed_lane_mask[labels == label] = 1

center_x = roi.shape[1] // 2
lookahead_y = 0.3



ys = []
left_pts = []
right_pts = []

visualise = np.zeros((roi.shape[0], roi.shape[1], 3), dtype=np.uint8)

for y in range(roi.shape[0]):
    xs = np.where(solid_lane_mask[y] > 0)[0]
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


if center_fit is not None and np.ndim(center_fit) == 1:
    target_x = np.polyval(center_fit, lookahead_y)
    steering_error = target_x - center_x
else:
    # Fallback: keep straight / last known center
    target_x = center_x
    steering_error = 0

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



vis = np.zeros((roi.shape[0], roi.shape[1], 3), dtype=np.uint8)

colors = np.random.randint(0, 255, (num_labels, 3))

for i in range(1, num_labels):
    vis[labels == i] = colors[i]

cv2.imshow("Connected Lines Components", vis)
cv2.imshow("Only Solid mask", solid_lane_mask * 255)
cv2.imshow("Only Dashed mask", dashed_lane_mask * 255)


# Draw Lookahead
y_px = int(lookahead_y * roi.shape[0])
cv2.circle(visualise, (int(target_x), y_px), 5, (0, 255, 255), -1)
print(f"Steering Error: {steering_error}")
        

cv2.line(visualise, (center_x, 0), (center_x, roi.shape[0]), (255, 255, 255), 1)
cv2.imshow("Lane", visualise)
cv2.imshow("Lane2", roi * 255)
cv2.waitKey(0)
