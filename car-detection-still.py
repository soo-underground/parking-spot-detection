import os
import numpy as np
import cv2
import mrcnn.config
import mrcnn.utils
from mrcnn.model import MaskRCNN
from pathlib import Path


# Configuration that will be used by the Mask-RCNN library
class MaskRCNNConfig(mrcnn.config.Config):
    NAME = "coco_pretrained_model_config"
    IMAGES_PER_GPU = 1
    GPU_COUNT = 1
    NUM_CLASSES = 1 + 80  # COCO dataset has 80 classes + one background class
    DETECTION_MIN_CONFIDENCE = 0.6


# Filter a list of Mask R-CNN detection results to get only the detected cars / trucks
def get_car_boxes(boxes, class_ids):
    car_boxes = []

    for i, box in enumerate(boxes):
        # If the detected object isn't a car / truck, skip it
        if class_ids[i] in [3, 8, 6]:
            car_boxes.append(box)

    return np.array(car_boxes)


# Root directory of the project
ROOT_DIR = Path(".")

# Directory to save logs and trained model
MODEL_DIR = os.path.join(ROOT_DIR, "logs")

# Local path to trained weights file
COCO_MODEL_PATH = os.path.join(ROOT_DIR, "mask_rcnn_coco.h5")

# Download COCO trained weights from Releases if needed
if not os.path.exists(COCO_MODEL_PATH):
    mrcnn.utils.download_trained_weights(COCO_MODEL_PATH)

# Directory of images to run detection on
IMAGE_DIR = os.path.join(ROOT_DIR, "images")

# Video file or camera to process - set this to 0 to use your webcam instead of a video file
VIDEO_SOURCE = "screen-capture.mp4"

# Create a Mask-RCNN model in inference mode
model = MaskRCNN(mode="inference", model_dir=MODEL_DIR, config=MaskRCNNConfig())

# Load pre-trained model
model.load_weights(COCO_MODEL_PATH, by_name=True)

# Location of parking spaces
parked_car_boxes = None

# Load the video file we want to run detection on
video_capture = cv2.VideoCapture(VIDEO_SOURCE)

img = cv2.imread('sources/kupchino.png')
rgb_image = img[:, :, ::-1]
results = model.detect([rgb_image], verbose=0)

# Mask R-CNN assumes we are running detection on multiple images.
# We only passed in one image to detect, so only grab the first result.
r = results[0]

# The r variable will now have the results of detection:
# - r['rois'] are the bounding box of each detected object
# - r['class_ids'] are the class id (type) of each detected object
# - r['scores'] are the confidence scores for each detection
# - r['masks'] are the object masks for each detected object (which gives you the object outline)

# Filter the results to only grab the car / truck bounding boxes
car_boxes = get_car_boxes(r['rois'], r['class_ids'])
f= open("spots.txt","w+")
print("Cars found in frame of video:")

# Draw each box on the frame
for box in car_boxes:
    print("Car: ", box)
    f.write(str(box))
    y1, x1, y2, x2 = box

    # Draw the box
    cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 1)

# Show the frame of video on the screen
f.close()
cv2.imshow('image', img)
cv2.waitKey(0)
cv2.destroyAllWindows()
