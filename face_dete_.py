import cv2
import time
import numpy as np
import pyttsx3
import threading
import pyrealsense2 as rs

# Load the COCO class names
with open("D:\OpenCV\object_detection_classes_coco.txt", 'r') as f:
    class_names = f.read().split('\n')

# Get a different color array for each of the classes
COLORS = np.random.uniform(0, 255, size=(len(class_names), 3))

# Load the DNN model
model = cv2.dnn.readNet(model="D:/OpenCV/frozen_inference_graph.pb",
                        config="D:\OpenCV\ssd_mobilenet_v2_coco_2018_03_29.pbtxt.txt",
                        framework='TensorFlow')
 
# Function to rescale frame
class DepthMap:
    def __init__(self, img_left, img_right):
        self.img_left = img_left
        self.img_right = img_right
        self.disparity = None

    def compute_depth_map_sgbm(self):
        min_disp = 0
        num_disparities = 128
        block_size = 15

        stereo = cv2.StereoSGBM_create(minDisparity=min_disp,
                                      numDisparities=num_disparities,
                                      blockSize=block_size,
                                      P1=8 * 3 * block_size ** 2,
                                      P2=32 * 3 * block_size ** 2)

        # Compute disparity map
        self.disparity = stereo.compute(self.img_left, self.img_right).astype(np.float32) / 16.0

    def compute_depth(self, x, y, focal_length, baseline):
        if self.disparity is None:
            print("Disparity map not computed.")
            return -1

        # Define the size of the square
        square_size = 5

        # Calculate the coordinates of the square
        x_start = max(0, x - square_size // 2)
        x_end = min(self.disparity.shape[1], x + square_size // 2)
        y_start = max(0, y - square_size // 2)
        y_end = min(self.disparity.shape[0], y + square_size // 2)

        # Extract the square region from the disparity map
        square_region = self.disparity[y_start:y_end, x_start:x_end]

        # Compute the average disparity value within the square region
        average_disparity = np.mean(square_region)

        if average_disparity <= 0:
            # print("Invalid disparity value.")
            return -1

        # Compute the depth using the average disparity
        depth = (focal_length * baseline) / average_disparity
        return depth*1000


def rescaleFrame(frame, scale=0.4):
    width = int(frame.shape[1] * scale)
    height = int(frame.shape[0] * .6)
    dimensions = (width, height)
    return cv2.resize(frame, dimensions, interpolation=cv2.INTER_AREA)

# Function to perform text-to-speech
import pyttsx3

def text_to_speech(detected, detected_objects):
    text_speech = pyttsx3.init()
    text_speech.setProperty('rate', 130)
    for class_name, depth in detected.items():
        if class_name not in detected_objects or detected_objects[class_name] ==-1:
            detected_objects[class_name] = depth
            text_speech.say(f"{class_name} is detected")
            if depth > 0:
                text_speech.say(f"at a distance of {depth} metres")
            else:
                text_speech.say("at an unknown distance")
            text_speech.runAndWait()

            

# Capture the video
# Final_video = cv2.VideoCapture(0)
# address = "https://192.168.188.42:8080/video"
# cap.open(address)
pipeline = rs.pipeline()
config = rs.config()
config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)

# Enable left and right infrared streams
config.enable_stream(rs.stream.infrared, 1)  # Left camera
config.enable_stream(rs.stream.infrared, 2)  # Right camera

# Start streaming
pipeline.start(config)
# Get the video frames' width and height for proper saving of videos
# frame_width = int(Final_video.get(3))
# frame_height = int(Final_video.get(4))
# desired_width = 640
# desired_height = 480

# Capture the video with the desired resolution
# cap.set(cv2.CAP_PROP_FRAME_WIDTH, desired_width)
# cap.set(cv2.CAP_PROP_FRAME_HEIGHT, desired_height)

# Create the `VideoWriter()` object
# out = cv2.VideoWriter('video_result.mp4', cv2.VideoWriter_fourcc(*'mp4v'), 30, (frame_width, frame_height))

# dictionary to keep track of detected objects in each frame of the video
detected_objects={}
while True:
    frames = pipeline.wait_for_frames()
    color_frame = frames.get_color_frame()

        # Convert color frame to a numpy array
    image = np.asanyarray(color_frame.get_data())
    left_frame = frames.get_infrared_frame(1)
    right_frame = frames.get_infrared_frame(2)

    if not left_frame or not right_frame:
        continue

    # Convert RealSense frames to numpy arrays
    left_image = np.asanyarray(left_frame.get_data())
    right_image = np.asanyarray(right_frame.get_data())

    # Create DepthMap instance
    # depth_map = DepthMap(left_image, right_image)

    # # Compute and display depth map using StereoSGBM
    # depth_map.compute_depth_map_sgbm()

    # # Example: Compute depth at pixel (443, 374)
    # depth = depth_map.compute_depth(443, 374, focal_length=1.93, baseline=50)

    # if depth is not None:
    #     print("Depth at pixel (443, 374):", depth)


    # Convert grayscale image to RGB
    # image = cv2.cvtColor(left_frame, cv2.COLOR_GRAY2RGB)

    # frame=rescaleFrame(frame)
    image_height, image_width, _ = image.shape
    blob = cv2.dnn.blobFromImage(image=image, size=(300, 300), mean=(104, 117, 123), swapRB=True)
    start_frame = time.time()
    model.setInput(blob)
    output = model.forward()
    end_frame = time.time()
    fps = 1/(end_frame - start_frame)

    # Reset detected objects list for each frame
    detected= {}

    for detection in output[0, 0, :, :]:
        confidence = detection[2]
        if confidence > .4:
            class_id = int(detection[1])
            class_name = class_names[class_id - 1]
            color = COLORS[class_id]
            box_x = detection[3] * image_width
            box_y = detection[4] * image_height
            box_width = detection[5] * image_width
            box_height = detection[6] * image_height
            depth_map = DepthMap(left_image, right_image)

            # Compute and display depth map using StereoSGBM
            depth_map.compute_depth_map_sgbm()

            # Example: Compute depth at pixel (443, 374)
            depth = depth_map.compute_depth(int((box_x + box_width) / 2), int((box_y + box_height) / 2), focal_length=.00193, baseline=.05)
            depth = round(depth, 2)
                
            # Add detected object to the dictionary
            detected[class_name]=depth
            cv2.rectangle(image, (int(box_x), int(box_y)), (int(box_width), int(box_height)), color, thickness=2)
            cv2.putText(image, class_name, (int(box_x), int(box_y - 5)), cv2.FONT_HERSHEY_PLAIN, 1, color, 2)
            cv2.putText(image, f"{round(depth,3)} m", (int(box_width-100), int(box_y - 5)), cv2.FONT_HERSHEY_PLAIN, 1, color, 2)
            
            # cv2.putText(image, f"{depth} ft", (20, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            # Create DepthMap instance
            
            # round depth to 2 decimal places
            

    cv2.imshow('image', image)

    # Start a new thread for text-to-speech
    threading.Thread(target=text_to_speech, args=(detected,detected_objects)).start()

    if cv2.waitKey(10) & 0xFF == ord('q'):
        break


# Final_video.release()
cv2.destroyAllWindows()
