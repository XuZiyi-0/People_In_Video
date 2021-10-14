import os

from general_config import general_cfg
from .common import keypoint_detection_common_cfg

mmpose_topdown_cfg = keypoint_detection_common_cfg.clone()

mmpose_topdown_cfg.PROJECT_ROOT = '/home/nk/mmpose/'
mmpose_topdown_cfg.POSE_CONFIG = os.path.join(mmpose_topdown_cfg.PROJECT_ROOT, 'configs/body/2d_kpt_sview_rgb_img/topdown_heatmap/coco/hrnet_w48_coco_256x192.py')
mmpose_topdown_cfg.POSE_CHECKPOINT = os.path.join(mmpose_topdown_cfg.PROJECT_ROOT, 'checkpoints/hrnet_w48_coco_256x192-b9e0b3ab_20200708.pth')
mmpose_topdown_cfg.DEVICE = 'cuda:0'
# mmpose_cfg.img_path = os.path.join(mmpose_cfg.PROJECT_ROOT, 'test_img')
# mmpose_cfg.img_names = os.listdir(mmpose_cfg.img_path)
mmpose_topdown_cfg.test_pipline =  test_pipeline = [
            dict(type='LoadImageFromFile'),
            dict(type='TopDownAffine'),
            dict(type='ToTensor'),
            dict(
                type='NormalizeTensor',
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]),
            dict(
                type='Collect',
                keys=['img'],
                meta_keys=[
                    'image_file', 'center', 'scale', 'rotation', 'bbox_score',
                    'flip_pairs'
                ]),
        ]
mmpose_topdown_cfg.skeleton = skeleton = [[15, 13], [13, 11], [16, 14], [14, 12], [11, 12],
                    [5, 11], [6, 12], [5, 6], [5, 7], [6, 8], [7, 9],
                    [8, 10], [1, 2], [0, 1], [0, 2], [1, 3], [2, 4],
                    [3, 5], [4, 6]]
mmpose_topdown_cfg.image_size = [192, 256]

