if __name__ == '__main__':
	# 将项目根路径添加到环境变量
	import os, sys
	sys.path.insert(0,os.path.abspath('../../'))

########################################################################################################################
from model_components.configs.pose_est_2d_img import keypoint_detection_common_cfg
from model_components.pose_est_2d_img.modeling import build_keypoint_detection_model

# Detection模块：
# 输入:
# img:list, img[i]:numpy.ndarray（(h,w,c)BGR顺序）
# 输出:
# det:list, ret[i]:tensor(n,6)  即n个检测框，每一行内容为：[x1, y1, x2, y2, confidence, class]
class keypoint_det:
	def __init__(self, cfg=build_keypoint_detection_model().clone()):
		self.cfg = cfg
		self.cfg.freeze()
		self.model = build_keypoint_detection_model(self.cfg)

	def run(self, img):
		det = self.model.run()
		return det
########################################################################################################################

# if __name__ == '__main__':
# 	# 单元测试：
# 	import cv2
# 	detection = Detection()
# 	img = cv2.imread(os.path.abspath('../../test_data/imgs/c1s1_032051.jpg'))
# 	det = detection.run([img])
keypoint_det.run()