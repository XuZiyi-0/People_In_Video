if __name__ == '__main__':
	# 将项目根路径添加到环境变量
	import os, sys
	sys.path.insert(0,os.path.abspath('../../'))

########################################################################################################################
from model_components.configs.detection import detection_common_cfg
from model_components.detection.modeling import build_det_model

# Detection模块：
# 输入:
# img:list, img[i]:numpy.ndarray（(h,w,c)BGR顺序）
# 输出:
# det:list, ret[i]:tensor(n,6)  即n个检测框，每一行内容为：[x1, y1, x2, y2, confidence, class]
class Detection:
	def __init__(self, cfg=detection_common_cfg.clone()):
		self.cfg = cfg
		self.cfg.freeze()
		self.model = build_det_model(self.cfg)

	def run(self, img):
		det = self.model.run(img)
		return det
########################################################################################################################

if __name__ == '__main__':
	# 单元测试：
	import cv2
	detection = Detection()
	img = cv2.imread(os.path.abspath('../../test_data/imgs/c1s1_032051.jpg'))
	det = detection.run([img])