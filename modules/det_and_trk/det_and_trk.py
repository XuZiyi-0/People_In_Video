import cv2

import sys, os
sys.path.insert(0, os.path.abspath('../../'))

from model_components.detection.detection import Detection
from model_components.tracking.tracking import Tracking
from modules.data_reader.general import VideoReader

class Detection_Tracking:
	def __init__(self, det_cfg=None, trk_cfg=None):
		if det_cfg:
			self.detection = Detection(det_cfg)
		else:
			self.detection = Detection()
		if trk_cfg:
			self.tracking = Tracking(trk_cfg)
		else:
			self.tracking = Tracking()

	def run(self, frame):
		bboxs, det_img_shape = self.detection.run(frame)

		return self.tracking.run(bboxs[0], frame, det_img_shape=det_img_shape, frame_shape=frame.shape)




if __name__ == '__main__':
	det_and_trk = Detection_Tracking()
	frame_reader = VideoReader()
	frame_reader.capture('/home/xzy/projects/Worksite_Monitors/test_data/2021湖人vs勇士/videos/clip0.mp4')
	# frame_reader.capture('/home/xzy/projects/Worksite_Monitors/test_data/videos/0.mp4')
	ret, frame = frame_reader.read()
	idx = 0
	while ret:
		tracklets, confs = det_and_trk.run(frame)
		print(idx)
		print(tracklets)

		print(confs)
		print('#'*64)
		idx += 1
		ret, frame = frame_reader.read()