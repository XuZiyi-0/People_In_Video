import os

from general_config import general_cfg
from .common import person_ReID_video_common_cfg

hypergraph_reid_cfg = person_ReID_video_common_cfg.clone()
hypergraph_reid_cfg.CHECKPOINT = os.path.join(general_cfg.PROJECT_ROOT, 'checkpoint/checkpoint_ep325.pth.tar')
hypergraph_reid_cfg.IMG_SIZE = (256,128)
hypergraph_reid_cfg.NUM_CLASS = 625 # MARS