import os
from fvcore.common.config import CfgNode

_C = CfgNode()  #defination
_C.MODEL = CfgNode()
_C.MODEL.NAME = 'build_byte'

_C.EXPN = False
_C.NAME = False
_C.CAMID = 0
_C.SAVE_RESULT = False
_C.EXP_FILE ='model_components/checkpoints/e2e_tracking//bytes/example/mot/yolox_x_mix_det.py'
_C.CHECKPOINT = os.path.abspath("model_components/checkpoints/e2e_tracking/bytes/bytetrack_x_mot20.tar")
_C.DEVICE = 0


_C.FRAME_RATE = 30
_C.CONF_THRES = False
_C.T_SIZE = False
_C.NMS_THRES = False
_C.FP16 = False
_C.FUSE = False
_C.TRT = False
_C.TRACK_THRES = 0.5
_C.TRACK_BUFFER = 30
_C.MATCH_THRES = 0.8
_C.MIN_BOX_AREA = 10
_C.MOT20 = False