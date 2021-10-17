import os
from fvcore.common.config import CfgNode

_C = CfgNode()  #defination
_C.MODEL = CfgNode()
_C.MODEL.NAME = 'build_jde'
_C.CFG ='model_components/checkpoints/e2e_tracking/yolov3_1088x608.cfg'
_C.CHECKPOINT = os.path.abspath("model_components/checkpoints/e2e_tracking/jde.uncertainty.pt")
_C.DEVICE = 0


_C.FRAME_RATE = 30
_C.CONF_THRES = 0.5
_C.IOU_THRES = 0.5
_C.NMS_THRES = 0.4
_C.MIN_BOX_AREA = 200
_C.TRACK_BUFFER =30
_C.OUTPUT_FORMAT='video'
_C.W = 1088
_C.H = 608
_C.IMG_SIZE=[1088, 608]