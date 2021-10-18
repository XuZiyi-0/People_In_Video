import os

from general_config import general_cfg
from .common import person_ReID_img_common_cfg

osnet_cfg = person_ReID_img_common_cfg.clone()

osnet_cfg.CHECKPOINT = os.path.join(general_cfg.PROJECT_ROOT, 'model_components/checkpoints/person_ReID_img/model.pth.tar-250')
osnet_cfg.IMG_SIZE = (256, 128)  # 256x128
osnet_cfg.Loss = "softmax"
osnet_cfg.NUM_CLASS = 751   # Market:751; Duke:702; VIPeR:316; CUHK01:871/486; CUHK03:1367/767
