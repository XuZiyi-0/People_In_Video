import cv2
import os, sys

sys.path.insert(0, os.path.abspath('../../'))
from model_components.configs.img_cls import img_cls_common_cfg
from model_components.img_cls.modeling import build_img_cls_model


class Img_cls:
	def __init__(self, cfg=img_cls_common_cfg.clone()):
		self.cfg = cfg
		self.cfg.freeze()
		self.model = build_img_cls_model(self.cfg)

	def run(self, img):
		return self.model.run(img)


if __name__ == '__main__':
	cfg = img_cls_common_cfg.clone()
	img_cls = Img_cls(cfg)

	img = cv2.imread('/home/gyj/resnet/data/cat.0.jpg')

	pred = img_cls.run(img)

	print(pred)