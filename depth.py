import numpy as np
import cv2 as cv
from matplotlib import pyplot as plt
import os
import pyrealsense2 as rs

class DepthMap:
    def __init__(self, img_left, img_right):
        self.img_left = img_left
        self.img_right = img_right

    def compute_depth_map_sgbm(self):
        # Adjustable parameters
        min_disp = 0
        num_disparities = 128
        block_size = 15

        stereo = cv.StereoSGBM_create(minDisparity=min_disp,
                                      numDisparities=num_disparities,
                                      blockSize=block_size,
                                      P1=8 * 3 * block_size ** 2,
                                      P2=32 * 3 * block_size ** 2)

        # Compute disparity map
        disparity = stereo.compute(self.img_left, self.img_right).astype(np.float32) / 16.0

        # Display the disparity mapxx
        plt.imshow(disparity, 'gray')
        plt.title('Disparity Map (StereoSGBM)')
        plt.colorbar()
        plt.show()

        # Compute depth map from disparity
        depth_map = self.compute_depth(disparity)

        # Display the depth map
        plt.imshow(depth_map, cmap='jet')
        plt.title('Depth Map')
        plt.colorbar()
        plt.show()

    def compute_depth(self, disparity):
        # Parameters
        baseline = 0.05  # Baseline in meters (50 mm)
        focal_length = 0.00193  # Focal length in meters (1.93 mm)

        # Compute depth from disparity map
        depth_map = (baseline * focal_length) / disparity
        return depth_map

# Configure RealSense pipeline
pipeline = rs.pipeline()
config = rs.config()

# Enable left and right infrared streams
config.enable_stream(rs.stream.infrared, 1)  # Left camera
config.enable_stream(rs.stream.infrared, 2)  # Right camera

# Start streaming
pipeline.start(config)

try:
    while True:
        # Wait for a new frame
        frames = pipeline.wait_for_frames()

        # Extract left and right infrared frames
        left_frame = frames.get_infrared_frame(1)
        right_frame = frames.get_infrared_frame(2)

        if not left_frame or not right_frame:
            continue

        # Convert RealSense frames to numpy arrays
        left_image = np.asanyarray(left_frame.get_data())
        right_image = np.asanyarray(right_frame.get_data())

        # Create DepthMap instance
        depth_map = DepthMap(left_image, right_image)

        # Compute and display depth map using StereoSGBM
        depth_map.compute_depth_map_sgbm()

        if cv.waitKey(1) & 0xFF == ord('x'):
            print("Exiting...")
            break

finally:
    # Stop streaming
    pipeline.stop()
    cv.destroyAllWindows()
