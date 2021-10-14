import os

from general_config import general_cfg
from .common import img_cls_common_cfg

resnet50_cfg = img_cls_common_cfg.clone()
resnet50_cfg.IMG_SIZE = (256,256)
resnet50_cfg.CHECKPOINT = os.path.join(general_cfg.PROJECT_ROOT, "model_components/checkpoints/img_cls/ep0_acc:50.40064102564102_loss:0.7052158453525641.pkl")
