from model_components.configs.pose_est_2d_img import keypoint_detection_common_cfg
from model_components.pose_est_2d_img.modeling import build_keypoint_detection_model
import cv2
from general_config import general_cfg
import sys
import os
sys.path.insert(0, os.path.abspath(general_cfg.PROJECT_ROOT + '/model_components/opensources/mmpose'))
# pose_est模块：
# 输入:
# img:list, img[i]:numpy.ndarray
# 输出:
# det:ndarray,
class keypoint_det:
    def __init__(self, cfg=keypoint_detection_common_cfg.clone()):
        self.cfg = cfg
        self.cfg.freeze()
        self.model = build_keypoint_detection_model(self.cfg)

    def run(self, img):
        det, heatmap = self.model.run(img)
        return det, heatmap
########################################################################################################################

#
#
if __name__ == '__main__':
    img_path ="/home/nk/mmpose/test_img/0_14.jpg"
    img = [cv2.imread(img_path)]
    # print(img)
    # print(img)
    det = keypoint_det()
    pred, heatmap = det.run(img)
    print(pred)
