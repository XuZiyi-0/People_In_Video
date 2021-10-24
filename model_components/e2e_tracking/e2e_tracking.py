import cv2
import torch
import sys, os
import numpy as np

sys.path.insert(0, os.path.abspath('../dec_and_trc/modeling'))
from model_components.configs.e2e_tracking.defaults_ByteTrack import _C
from model_components.e2e_tracking.modeling.build import build_det_and_trc_model
from model_components.e2e_tracking.modeling.jde.utils.datasets import letterbox


class Detection_and_Tracking:
    def __init__(self, cfg=_C.clone()):
        self.cfg = cfg

        self.model = build_det_and_trc_model(self.cfg)

    def run(self, img):
        return self.model.run(img)


if __name__ == '__main__':
    video = cv2.VideoCapture('E://reid//JDE//input//ex.mp4')

    dec_and_trc = Detection_and_Tracking()

    for i in range(10):
        res, frame = video.read()

        img0 = frame

        online_targets = dec_and_trc.run(img0)

        for t in online_targets:
            tlwh = t.tlwh
            tid = t.track_id
            print(tid, tlwh)
