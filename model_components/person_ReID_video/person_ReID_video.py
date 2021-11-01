if __name__ == '__main__':
	import os, sys
	import warnings
	warnings.filterwarnings("ignore")

	sys.path.insert(0, os.path.abspath('../../'))

########################################################################################################################
from model_components.configs.person_ReID_video import person_ReID_video_common_cfg
from model_components.person_ReID_video.modeling import build_person_reid_video_model

# PersonReIDImg模块：
# 输入:
# imgs:list, imgs[i]:numpy.ndarray（(h,w,c)BGR顺序）
# 输出:
# features:tensor(n,m)    即输入的n张图片的m维特征
class PersonReIDVideo:
	def __init__(self, cfg=person_ReID_video_common_cfg.clone()):
		self.cfg = cfg
		self.cfg.freeze()
		self.model = build_person_reid_video_model(self.cfg)

	def run(self, imgs):
		features = self.model.run(imgs)
		return features
########################################################################################################################

if __name__ == '__main__':
	# 单元测试
	import cv2
	person_reid_video = PersonReIDVideo()
	imgs = []
	imgs_path = '/home/mnx/datasets/MARS-v160809/bbox_test/0010'
	img_names = os.listdir(imgs_path)
	for img_name in img_names:
		imgs.append(cv2.imread(os.path.join(imgs_path, img_name)))
	features = person_reid_video.run(imgs)
	print(features)