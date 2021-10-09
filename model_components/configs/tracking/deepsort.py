import os

from general_config import general_cfg
from .common import tracking_common_cfg

deepsort_cfg = tracking_common_cfg.clone()

deepsort_cfg.REID_CKPT = os.path.join(general_cfg.PROJECT_ROOT, 'model_components/checkpoints/tracking/deepsort.t7')
deepsort_cfg.MAX_DIST = 0.2
deepsort_cfg.MIN_CONFIDENCE = 0.3
deepsort_cfg.NMS_MAX_OVERLAP = 0.5
deepsort_cfg.MAX_IOU_DISTANCE = 0.7
deepsort_cfg.MAX_AGE = 70
deepsort_cfg.N_INIT = 3
deepsort_cfg.NN_BUDGET = 100
