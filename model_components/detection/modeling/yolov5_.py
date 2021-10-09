import torch
import numpy as np
import cv2

from model_components.configs.detection import yolov5_cfg
from model_components.detection.modeling.yolov5.models.experimental import attempt_load
from model_components.detection.modeling.yolov5.utils.general import non_max_suppression
from model_components.detection.modeling.yolov5.utils.datasets import letterbox
from model_components.detection.modeling.utils import scale_coords

from model_components.detection.modeling.build import DETECTION_REGISTRY


@DETECTION_REGISTRY.register()
def build_yolov5():
	return YoloV5()

class YoloV5:
	def __init__(self, cfg=yolov5_cfg.clone()):
		self.cfg = cfg
		self.cfg.freeze()
		self.device = torch.device('cuda:%d'%self.cfg.DEVICE_ID)
		self.model = attempt_load(cfg.CHECKPOINT, map_location='cuda:%d'%self.cfg.DEVICE_ID)
		self.model.half()  # to FP16

	def ndarray_BGR_img_pre_process(self, imgs0):
		imgs = []
		for i in range(len(imgs0)):
			img = letterbox(imgs0[i], self.cfg.IMG_SIZE, stride=self.cfg.STRIDE)[0]
			img = img[:, :, ::-1].transpose(2, 0, 1)
			img = np.ascontiguousarray(img)
			imgs.append(img)
		imgs = np.array(imgs)
		imgs = torch.from_numpy(imgs).to(self.device)
		imgs = imgs.half()
		imgs /= 255.0
		return imgs

	def run(self, imgs0):
		imgs = self.ndarray_BGR_img_pre_process(imgs0)
		pred = self.model(imgs)
		det = non_max_suppression(pred[0], self.cfg.CONF_THRES, self.cfg.IOU_THRES, classes=self.cfg.CLASSES, agnostic=self.cfg.AGNOSTIC_NMS)
		for i in range(len(det)):
			det[i][:, :4] = scale_coords(imgs.size()[2:], det[i][:, :4], imgs0[i].shape).round()
		return det

if __name__ == '__main__':
	# 单元测试
	import os, sys
	sys.path.insert(0, os.path.abspath('../../../'))
	detector = build_yolov5()
	img = cv2.imread(os.path.abspath('../../../test_data/imgs/c1s1_032051.jpg'))
	pred = detector.run([img])
	print(pred)