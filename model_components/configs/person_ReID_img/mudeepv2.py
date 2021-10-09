import os

from general_config import general_cfg
from .common import person_ReID_img_common_cfg

mudeepv2_cfg = person_ReID_img_common_cfg.clone()

mudeepv2_cfg.CHECKPOINT = os.path.join(general_cfg.PROJECT_ROOT, 'model_components/checkpoints/person_ReID_img/mudeepv2_market.pkl')
mudeepv2_cfg.IMG_SIZE = (384, 192)  # 384 x 128
mudeepv2_cfg.NUM_CLASS = 751   # Market:751; Duke:702; VIPeR:316; CUHK01:871/486; CUHK03:1367/767
