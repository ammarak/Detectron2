from detectron2.engine import DefaultPredictor
from detectron2.config import get_cfg
from detectron2.data import MetadataCatalog
from detectron2.utils.visualizer import ColorMode, Visualizer
from detectron2 import model_zoo


import cv2
import numpy as np

import os
os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"

import warnings
warnings.filterwarnings("ignore")

# Instance Segmentation
cfg = get_cfg()

# Load the model config and pre-trained model
cfg.merge_from_file(model_zoo.get_config_file("COCO-InstanceSegmentation/mask_rcnn_R_101_FPN_3x.yaml"))
cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url("COCO-InstanceSegmentation/mask_rcnn_R_101_FPN_3x.yaml")

cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.7
cfg.MODEL.DEVICE = "cuda"


predictor = DefaultPredictor(cfg)

image = cv2.imread("horse.jpeg")
# cv2.imshow('Image', image)
# cv2.waitKey(0)
predictions = predictor(image)

viz = Visualizer(image[:,:,::-1], 
                 metadata = MetadataCatalog.get(cfg.DATASETS.TRAIN[0]),
                 instance_mode = ColorMode.IMAGE
                 )

output = viz.draw_instance_predictions(predictions['instances'].to("cpu"))

#cv2.imshow('Image', image)
cv2.imshow('Results', output.get_image()[:,:,::-1])
cv2.waitKey(0)

print("Done")
