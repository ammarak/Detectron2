from detectron2.engine import DefaultPredictor
from detectron2.config import get_cfg
from detectron2.data import MetadataCatalog
from detectron2.utils.visualizer import ColorMode, Visualizer
from detectron2 import model_zoo

from imutils.video import FPS

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
cfg.MODEL.DEVICE = "cpu"


detector = DefaultPredictor(cfg)

cap = cv2.VideoCapture("General_public_preview.mp4")

fps = FPS().start()

while cap.isOpened():
    ret, frame = cap.read()
    print(f"{cap.get(1)}/{cap.get(cv2.CAP_PROP_FRAME_COUNT)}")
    if not ret:
        break

    # Perform object detection with Detectron2
    outputs = detector(frame)

    # Get boxes, classes, and scores from the detection results
    boxes = outputs["instances"].pred_boxes.tensor.cpu().numpy()
    classes = outputs["instances"].pred_classes.cpu().numpy()
    scores = outputs["instances"].scores.cpu().numpy()

    # Loop through detected objects and overlay bounding boxes
    for box, class_id, score in zip(boxes, classes, scores):
        x1, y1, x2, y2 = map(int, box)
        class_name = MetadataCatalog.get(cfg.DATASETS.TRAIN[0]).thing_classes[class_id]
        label = f"{class_name}: {score:.2f}"
        color = (0, 255, 0)  # Green color
        thickness = 2
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, thickness)
        cv2.putText(frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, thickness)

    # Display the frame with bounding boxes
    cv2.imshow('Detectron2 Video', frame)
    
    fps.update()
    
    # Check for user interrupt (press 'q' to exit)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

fps.stop()
print("Elasped time: {:.2f}".format(fps.elapsed()))
print("FPS: {:.2f}".format(fps.fps()))
    
# Release the video objects
cap.release()
cv2.destroyAllWindows()

