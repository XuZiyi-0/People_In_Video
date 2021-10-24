import torch
import numpy as np
import cv2
import os, sys
sys.path.insert(0, os.path.abspath('../../dec_and_trc/modeling/ByteTrack'))

from model_components.configs.e2e_tracking.defaults_ByteTrack import _C
from model_components.e2e_tracking.modeling.ByteTrack.tools.demo_track import Predictor
from model_components.e2e_tracking.modeling.ByteTrack.yolox.exp.build import get_exp,get_exp_by_file,get_exp_by_name
from model_components.e2e_tracking.modeling.ByteTrack.yolox.utils import fuse_model,get_model_info,postprocess,vis
from model_components.e2e_tracking.modeling.ByteTrack.yolox.tracker.byte_tracker import BYTETracker
from model_components.e2e_tracking.modeling.build import DET_AND_TRC_REGISTRY

from model_components.e2e_tracking.modeling.ByteTrack.yolox.tracking_utils.timer import Timer
import time

@DET_AND_TRC_REGISTRY.register()
def build_byte(cfg):
    return Byte(cfg)

class Byte:
    def __init__(self,cfg=_C.clone()):
        torch.cuda.set_device(0)
        self.cfg=cfg

        self.exp = get_exp(cfg.EXP_FILE,cfg.NAME)

        #print(self.exp)
        if not cfg.EXPN:
            cfg.EXPN = self.exp.exp_name
        file_name = os.path.join(self.exp.output_dir, cfg.EXPN)
        os.makedirs(file_name, exist_ok=True)

        if cfg.SAVE_RESULT:
            vis_folder = os.path.join(file_name, "track_vis")
            os.makedirs(vis_folder, exist_ok=True)

        if cfg.TRT:
            cfg.DEVICE = "gpu"

        if cfg.CONF_THRES:
            self.exp.test_conf = cfg.CONF_THRES
        if cfg.NMS_THRES:
            self.exp.nmsthre = cfg.NMS_THRES
        if cfg.T_SIZE:
            self.exp.test_size = (cfg.T_SIZE, cfg.T_SIZE)

        model = self.exp.get_model()
        if cfg.DEVICE == "gpu":
            model.cuda()
        model.eval()

        if not cfg.TRT:
            if cfg.CHECKPOINT is None:
                ckpt_file = os.path.join(file_name, "best_ckpt.pth.tar")
            else:
                ckpt_file = cfg.CHECKPOINT

            ckpt = torch.load(ckpt_file, map_location="cpu")
            # load the model state dict
            model.load_state_dict(ckpt["model"])

        if cfg.FUSE:

            model = fuse_model(model)

        if cfg.FP16:
            model = model.half()  # to FP16

        if cfg.TRT:
            assert not cfg.FUSE, "TensorRT model is not support model fusing!"
            trt_file = os.path.join(file_name, "model_trt.pth")
            assert os.path.exists(
                trt_file
            ), "TensorRT model is not found!\n Run python3 tools/trt.py first!"
            model.head.decode_in_inference = False
            decoder = model.head.decode_outputs

        else:
            trt_file = None
            decoder = None
        self.predictor=Predictor(model,self.exp,trt_file,decoder,cfg.DEVICE,cfg.FP16)
        self.tracker = BYTETracker(cfg, frame_rate=30)


    def run(self,img0):


        timer= Timer()
        outputs, img_info = self.predictor.inference(img0, timer)
        online_targets = self.tracker.update(outputs[0], [img_info['height'], img_info['width']], self.exp.test_size)



        return online_targets


if __name__ == '__main__':
    video = cv2.VideoCapture('E://reid//JDE//input//ex.mp4')

    print(type(video))
    dec_and_trc = build_byte(_C.clone())

    for i in range(10):
        res, frame = video.read()

        img0 = frame

        online_targets = dec_and_trc.run(img0)

        for t in online_targets:
            tlwh = t.tlwh
            tid = t.track_id
            print(tid,tlwh)




