import numpy as np
import os
import cv2
import mrcnn.config
import mrcnn.utils
from mrcnn.model import MaskRCNN
from pathlib import Path

img = cv2.imread('sources/kupchino.png')
rgb_image = img[:, :, ::-1]



class MaskRCNNConfig(mrcnn.config.Config):
    NAME = "coco_pretrained_model_config"
    IMAGES_PER_GPU = 1
    GPU_COUNT = 1
    NUM_CLASSES = 1 + 80  # COCO dataset has 80 classes + one background class
    DETECTION_MIN_CONFIDENCE = 0.6
ROOT_DIR = Path(".")
MODEL_DIR = os.path.join(ROOT_DIR, "logs")
COCO_MODEL_PATH = os.path.join(ROOT_DIR, "mask_rcnn_coco.h5")
model = MaskRCNN(mode="inference", model_dir=MODEL_DIR, config=MaskRCNNConfig())
model.load_weights(COCO_MODEL_PATH, by_name=True)

spots = np.load('spots.npy')
for spot in spots:
    print(spot, '===')
    roi = img[spot[0]:spot[1], spot[2]:spot[3]]
    rgb_roi = roi[:, :, ::-1]
    results = model.detect([rgb_image], verbose=0)
    r = results[0]
    if 3 in r['class_ids'] :
        print('place taken')
    else:
        print('place vacant')
#cv2.imshow('image', img)
#cv2.waitKey(0)
#cv2.destroyAllWindows()
