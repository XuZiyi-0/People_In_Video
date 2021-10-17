import cv2
import torch
import sys, os
import numpy as np
sys.path.insert(0, os.path.abspath('../e2e_tracking/modeling/jde'))
from model_components.configs.e2e_tracking.defaults_jde import _C
from model_components.e2e_tracking.modeling.build import build_det_and_trc_model
from model_components.e2e_tracking.modeling.jde.utils.datasets import letterbox
class Detection_and_Tracking:
	def __init__(self, cfg=_C.clone()):
		self.cfg = cfg
		self.cfg.freeze()

		self.model = build_det_and_trc_model(self.cfg)

	def run(self, im_blob,img):
		return self.model.run(im_blob, img)

if __name__ == '__main__':
    video = cv2.VideoCapture('/home/xzy/projects/People_In_Video/test_data/videos/0.mp4')

    dec_and_trc = Detection_and_Tracking()

    for i in range(10):
        res, frame = video.read()

        img0 = frame

        img0 = cv2.resize(img0, (dec_and_trc.model.w, dec_and_trc.model.h))

        img, _, _, _ = letterbox(img0, height=dec_and_trc.model.h, width=dec_and_trc.model.w)

        img = img[:, :, ::-1].transpose(2, 0, 1)
        img = np.ascontiguousarray(img, dtype=np.float32)
        img /= 255.0

        blob = torch.from_numpy(img).cuda().unsqueeze(0)

        online_targets = dec_and_trc.run(blob, img0)


        for t in online_targets:
            tlwh = t.tlwh
            tid = t.track_id
            print(tid,tlwh)
