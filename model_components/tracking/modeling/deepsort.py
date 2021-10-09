import cv2

from model_components.configs.tracking import deepsort_cfg
from model_components.tracking.modeling.deep_sort.deep_sort import DeepSort
from model_components.tracking.modeling.utils import xyxy2xywh
from model_components.tracking.modeling.build import TRACKING_REGISTRY

@TRACKING_REGISTRY.register()
def build_deepsort():
	return DeepSort_()

class DeepSort_:
	def __init__(self, cfg=deepsort_cfg.clone()):
		self.cfg = cfg
		self.cfg.freeze()
		self.model = DeepSort(
						self.cfg.REID_CKPT,
						device_ID=cfg.DEVICE_ID,
                        max_dist=self.cfg.MAX_DIST,
                        min_confidence=self.cfg.MIN_CONFIDENCE,
                        nms_max_overlap=self.cfg.NMS_MAX_OVERLAP,
                        max_iou_distance=self.cfg.MAX_IOU_DISTANCE,
                        max_age=self.cfg.MAX_AGE,
						n_init=self.cfg.N_INIT,
						nn_budget=self.cfg.NN_BUDGET,
                        use_cuda=True
						)

	def run(self, det, img):
		xywhs = xyxy2xywh(det[:, 0:4])
		confs = det[:, 4]
		clss = det[:, 5]
		tracklets, infos = self.model.update(xywhs.cpu(), confs.cpu(), clss, img)
		return tracklets, infos

if __name__ == '__main__':
	# 单元测试：
	import time
	import os, sys
	sys.path.insert(0, os.path.abspath('../../../'))

	from model_components.detection.detection import Detection
	video = cv2.VideoCapture('/home/xzy/projects/People_In_Video/test_data/videos/0.mp4')
	detection = Detection()
	tracking = build_deepsort()
	t0 = time.time()
	n = 1000
	for i in range(n):
		res, frame = video.read()
		det = detection.run(frame)[0]
		tracklets, infos = tracking.run(det, frame)
		print(tracklets)
		for key in infos.keys():
			print(key,':')
			print(infos[key])
	t1 = time.time()
	print('\ntime_cost:  %.3fs/frame'%((t1-t0)/n))











