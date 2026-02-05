import numpy as np
import cv2
import carla
import torch
import math
from PIL import Image
from torchvision import transforms
from enet import ENet


class Lane_Decision:
    def __init__(self, broker):
        self.mask = None
        self.broker = broker

        self.broker.subscribe("road_mask", self.mask_to_decision)
        print("Lane_Decision class initialized")

    def road_mask_callback(self, mask):
        
        self.mask = mask

    def calculate_solid_boundary(self, x, y, h, w, center_x, roi):
        Y_NEAR_CAR = int(0.75 * roi.shape[0])   # Only look at components where the y is close to the car
        MARGIN = 10

        bottom_y = y + h
        if bottom_y < Y_NEAR_CAR:
            return 0.0

        comp_left = x
        comp_right = x + w

        # If component is on the left 
        if comp_right <= center_x:
            dist = (center_x - MARGIN) - comp_right

            # Dist < 0 means its intruding over the line
            if dist < 0:
                intrude = -dist
                return +intrude # Go right
            return 0.0
            

        elif comp_left >= center_x:
            dist = comp_left - (center_x + MARGIN)
            if dist < 0:
                intrude = -dist
                return -intrude # Go left
            return 0.0
        
        else:
            # Your ontop of the line
            overlap_left = center_x - comp_left
            overlap_right = comp_right - center_x

            if overlap_left > overlap_right:
                # More overlap on left side so go right
                error = +overlap_left
            else:
                # More overlap on right side so go left
                error = -overlap_right

        return error
        
    def component_angle(self, mask):
        ys, xs = np.where(mask > 0)
        if len(xs) < 20:
            return None
        pts = np.column_stack((xs.astype(np.float32), ys.astype(np.float32)))
        pts -= pts.mean(axis=0, keepdims=True)

        cov = np.cov(pts.T)
        eigvals, eigvecs = np.linalg.eig(cov)
        v = eigvecs[:, np.argmax(eigvals)] # Principal direction
        angle = math.degrees(math.atan2(v[1], v[0]))  
        return angle    # 0 Horizontal, 90 Vertical

    def mask_to_decision(self, mask):
        """ Runs everytime a new mask from the ENET comes in """
        self.mask = mask

        if self.mask is None:
            return

        LANE_CLASS = 2
        h, w = self.mask.shape

        # Extract the lane info
        lane_mask = (self.mask == LANE_CLASS).astype(np.uint8)
        roi = lane_mask[int(0.6 * h): h, :]

        # Define where the center and lookahead will be
        center_x = roi.shape[1] // 2
        lookahead_y = 0.3

        # roi is your binary lane mask in ROI (uint8 0/255)
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))

        # Connect small gaps (shadow breaks, patchy segmentation)
        roi_closed = cv2.morphologyEx(roi, cv2.MORPH_CLOSE, kernel, iterations=1)

        # Find Connected Parts of Lane Lines
        num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(roi_closed, connectivity=8)
        solid_lane_mask = np.zeros_like(roi)
        MIN_VERT_SPAN = int(0.5 * roi.shape[0])
        MIN_PIXEL = 200
        solid_components = []
        solid_line_error = 0.0 # Default value
        MAX_WIDTH = 45
        MAX_AREA_PER_H = 100
        MAX_FILL_RATIO = 0.35
        MIN_VERTICAL_ANGLE = 10.0

       
        # Now filter out small lines, thick blobs, etc...
        for i in range(1, num_labels): # Skip the background
            x, y, w, h, area = stats[i]
            
            # Too small = Reject
            if h < MIN_VERT_SPAN or area < MIN_PIXEL:
                continue
            
            # Thiness Check
            area_per_h = area / max(h, 1)
            fill_ratio = area / max(w * h, 1)

            # if w > MAX_WIDTH:
            #     continue
            if area_per_h > MAX_AREA_PER_H:
                print("MAX AREA PER H BLOCK")
                continue
            # if fill_ratio > MAX_FILL_RATIO:
            #     print("Fill Ratio Blocks")
            #     continue

           
            # Check Lines Orientation
            temp_mask = (labels == i).astype(np.uint8)
            angle = self.component_angle(temp_mask)
            if angle is None:
                print("ANGLE Blocks NONE")
                continue

            # Normalise Angle into [-90,90] and check if its vertical
            angle_normalised = ((angle + 90) % 180) - 90
            if abs(angle_normalised) < MIN_VERTICAL_ANGLE:
                print("ANGLE Blocks")
                continue

            # If we got here. Its a lane line
            solid_components.append(i)

            # Check or solids we are driving too close to
            solid_line_error += self.calculate_solid_boundary(x, y, h, w, center_x, roi)


        self.broker.publish("solid_intrusion", float(solid_line_error))
        # Add these big lines to the solid mask
        for label in solid_components:
            solid_lane_mask[labels == label] = 1
            

        
        


        # Define the pts of the closest left and right lines
        ys = []
        left_pts = []
        right_pts = []
        visualise = np.zeros((roi.shape[0], roi.shape[1], 3), dtype=np.uint8)

        # Find the closest
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

        # Fit the lines
        if len(ys) > 10:
            left_fit = np.polyfit(ys, left_pts, 2)
            right_fit = np.polyfit(ys, right_pts, 2)
            center_fit = (left_fit + right_fit) / 2
        else:
            left_fit = right_fit = center_fit = None

        # Calculate target and steering error 
        if center_fit is not None and np.ndim(center_fit) == 1:
            target_x = np.polyval(center_fit, lookahead_y)
            steering_error = target_x - center_x
        else:
            # Fallback: keep straight / last known center
            target_x = center_x
            steering_error = 0
        
        steering_error = float(np.clip(steering_error, -300, 300))  # Clamp Values 


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

        # Colour the lanes a different colour
        vis = np.zeros((roi.shape[0], roi.shape[1], 3), dtype=np.uint8)
        colors = np.random.randint(0, 255, (num_labels, 3))
        for i in range(1, num_labels):
            vis[labels == i] = colors[i]

        
        

        # Draw Lookahead
        y_px = int(lookahead_y * roi.shape[0])
        cv2.circle(visualise, (int(target_x), y_px), 5, (0, 255, 255), -1)
        #print(f"Steering Error: {steering_error}")

        cv2.line(visualise, (center_x, 0), (center_x, roi.shape[0]), (255, 255, 255), 1)


        display_size = (256, 256)
        resized_vis = cv2.resize(vis, display_size)
        resized_solid_mask = cv2.resize(solid_lane_mask, display_size)
        resized_visualise = cv2.resize(visualise, display_size)
        resized_roi = cv2.resize(roi, display_size)

        """
        cv2.imshow("Connected Lines Components", resized_vis)
        cv2.imshow("Only Solid mask", resized_solid_mask * 255)
        cv2.imshow("Steering", resized_visualise)
        cv2.imshow("ALL", resized_roi * 255)
        cv2.waitKey(1)
        """
        self.broker.publish("steering_error", steering_error)

        
        
        


