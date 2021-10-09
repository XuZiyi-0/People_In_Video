import os

from general_config import general_cfg
from .common import detection_common_cfg

yolov5_cfg = detection_common_cfg.clone()

yolov5_cfg.CHECKPOINT = os.path.join(general_cfg.PROJECT_ROOT, "model_components/checkpoints/detection/crowdhuman_yolov5m.pt")
yolov5_cfg.IMG_SIZE = (640, 640)
yolov5_cfg.STRIDE = 32
yolov5_cfg.CONF_THRES = 0.7
yolov5_cfg.IOU_THRES = 0.5
yolov5_cfg.CLASSES = (0)
yolov5_cfg.AGNOSTIC_NMS = False