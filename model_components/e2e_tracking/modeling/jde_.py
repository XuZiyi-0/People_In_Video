import torch
import numpy as np
import cv2
import os, sys
sys.path.insert(0, os.path.abspath('../../dec_and_trc/modeling/jde'))

from model_components.configs.e2e_tracking.defaults_jde import _C
from model_components.e2e_tracking.modeling.jde.utils.datasets import letterbox
from model_components.e2e_tracking.modeling.jde.tracker.multitracker import JDETracker

from model_components.e2e_tracking.modeling.build import DET_AND_TRC_REGISTRY

@DET_AND_TRC_REGISTRY.register()
def build_jde(cfg):
    return JDE(cfg)

class JDE:
    def __init__(self,cfg=_C.clone()):
        self.cfg=cfg
        self.cfg.freeze()
        self.w = self.cfg.W
        self.h = self.cfg.H
        self.img_size=self.cfg.IMG_SIZE
        self.device = torch.device('cuda:%d' % self.cfg.DEVICE)
        self.model = JDETracker(
            self.cfg.CFG,
            self.cfg.CHECKPOINT,
            self.cfg.IOU_THRES,
            self.cfg.NMS_THRES,
            self.cfg.CONF_THRES,
            self.cfg.MIN_BOX_AREA,
            self.cfg.TRACK_BUFFER,
            self.cfg.IMG_SIZE,
            frame_rate=30)

    def run(self,img0):
        img0 = cv2.resize(img0, (dec_and_trc.model.w, dec_and_trc.model.h))

        img, _, _, _ = letterbox(img0, height=dec_and_trc.model.h, width=dec_and_trc.model.w)

        img = img[:, :, ::-1].transpose(2, 0, 1)
        img = np.ascontiguousarray(img, dtype=np.float32)
        img /= 255.0

        blob = torch.from_numpy(img).cuda().unsqueeze(0)
        output=self.model.update(blob,img)
        return output


if __name__ == '__main__':
    video = cv2.VideoCapture('E://reid//JDE//input//ex.mp4')

    print(type(video))
    dec_and_trc = build_jde(_C.clone())

    for i in range(10):
        res, frame = video.read()

        img0 = frame

        online_targets = dec_and_trc.run(img0)


        for t in online_targets:
            tlwh = t.tlwh
            tid = t.track_id
            print(tid,tlwh)




