if __name__ == '__main__':
	# 将项目根路径添加到环境变量
	import os, sys
	sys.path.insert(0,os.path.abspath('../../'))

########################################################################################################################
from model_components.configs.tracking import tracking_common_cfg
from model_components.tracking.modeling import build_trk_model

# Tracking模块：
# 输入：
# det:tensor(n,6)  即n个检测框，每一行内容为：[x1, y1, x2, y2, confidence, class]
# img:numpy.ndarray(h,w,c)  BGR顺序
# 输出：
# tracklets:numpy.adarray(n,6)  即n个踪片，每一行内容为：[x1, y1, x2, y2, tracklet_id, class]
# info:dist    追踪模型中间数据，
class Tracking:
	def __init__(self, cfg=tracking_common_cfg.clone()):
		self.cfg = cfg
		self.cfg.freeze()
		self.model = build_trk_model(self.cfg)

	def run(self, det, img):
		tracklets, infos =  self.model.run(det, img)
		return tracklets, infos
########################################################################################################################

if __name__ == '__main__':
	# 单元测试
	import cv2
	from model_components.detection.detection import Detection
	video = cv2.VideoCapture('/home/xzy/projects/People_In_Video/test_data/videos/0.mp4')

	detection = Detection()

	tracking = Tracking()

	import time
	t0 = time.time()
	n = 10
	for i in range(n):
		res, frame = video.read()

		det = detection.run([frame])[0]

		tracklets, infos = tracking.run(det, frame)
		print(tracklets)
		print(type(tracklets))
		for key in infos.keys():
			print(key,':')
			print(infos[key])
	t1 = time.time()
	print('\ntime_cost:  %.3fs/frame'%((t1-t0)/n))
