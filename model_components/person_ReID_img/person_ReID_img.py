if __name__ == '__main__':
	import os, sys
	import warnings
	warnings.filterwarnings("ignore")

	sys.path.insert(0, os.path.abspath('../../'))

########################################################################################################################
from model_components.configs.person_ReID_img import person_ReID_img_common_cfg
from model_components.person_ReID_img.modeling import build_person_reid_img_model

# PersonReIDImg模块：
# 输入:
# imgs:list, imgs[i]:numpy.ndarray（(h,w,c)BGR顺序）
# 输出:
# features:tensor(n,m)    即输入的n张图片的m维特征
class PersonReIDImg:
	def __init__(self, cfg=person_ReID_img_common_cfg.clone()):
		self.cfg = cfg
		self.cfg.freeze()
		self.model = build_person_reid_img_model(self.cfg)

	def run(self, imgs):
		features = self.model.run(imgs)
		return features
########################################################################################################################

if __name__ == '__main__':
	# 单元测试
	import cv2
	person_reid_img = PersonReIDImg()
	img = cv2.imread(os.path.abspath('../../test_data/imgs/c1s1_032051.jpg'))
	features = person_reid_img.run([img])
	print(features)