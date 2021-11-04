import os
import torch
import cv2
import numpy as np
import time
from torchvision import transforms
from PIL import Image
import warnings

warnings.filterwarnings('ignore')

from model_components.person_ReID_video.modeling.hypergraph_reid.models.ResNet_hypergraphsage_part import \
    ResNet50GRAPHPOOLPARTHyper
from model_components.person_ReID_video.modeling.hypergraph_reid.models import resnet3d
from model_components.utils import BGR_hwc_to_RGB_chw
from model_components.configs.person_ReID_video import hypergraph_reid_cfg

from model_components.person_ReID_video.modeling.build import PERSON_REID_VIDEO_REGISTRY


@PERSON_REID_VIDEO_REGISTRY.register()
def build_hypergraph_reid():
    return hypergraph_reid_()


class hypergraph_reid_:
    def __init__(self, cfg=hypergraph_reid_cfg.clone()):
        self.cfg = cfg
        self.cfg.freeze()
        self.device = torch.device('cuda:%d' % self.cfg.DEVICE_ID[0])

        self.model = ResNet50GRAPHPOOLPARTHyper(pool_size=8, input_shape=2048, n_classes=self.cfg.NUM_CLASS,
                                                loss={'xent', 'htri'})
        self.model.to(self.device)
        # self.model = torch.nn.DataParallel(self.model, device_ids=cfg.DEVICE_ID)
        self.model.eval()
        with torch.no_grad():
            self.model.load_state_dict(torch.load(self.cfg.CHECKPOINT, map_location=self.device)['state_dict'])

        self.transformer = transforms.Compose([
            transforms.Resize(self.cfg.IMG_SIZE),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])

        self.imgs_list = []
        # self.img_root = '/home/mnx/datasets/MARS-v160809/bbox_test/0010'

    def fliplr(self, img):
        inv_idx = torch.arange(img.size(3) - 1, -1, -1).long()  # N x c x H x W
        img_flip = img.index_select(3, inv_idx)
        return img_flip


    def preprocess(self, clips0):
        clips = []
        for imgs0 in clips0:
            imgs = []
            for i in range(len(imgs0)):
                # print("aaa",imgs0[i])
                img = cv2.cvtColor(imgs0[i], cv2.COLOR_BGR2RGB)
                img = Image.fromarray(img)  # array转image,transform接收image格式的输入
                img = self.transformer(img)
                img = img.unsqueeze(0)
                imgs.append(img)
            imgs = torch.cat(imgs, dim=0)
            clips.append(imgs)
        clips = torch.stack(clips)
        return clips

    def run(self, clips):
        clips = self.preprocess(clips)
        clips = clips.to(self.device)
        with torch.no_grad():
            # b=1, n=number of clips, s=16
            n, s, c, h, w = clips.size()
            # assert (b == 1)
            # imgs = imgs.view(n, s, c, h, w)
            # features = model(imgs, adj1, adj2, adj3)
            features = self.model(clips)
            # 特征向量归一化
            fnorm = torch.norm(features, p=2, dim=1, keepdim=True)
            features = features.div(fnorm.expand_as(features))
            features = features.data.cpu()
        return features


if __name__ == '__main__':
    # 单元测试
    clips_path0 = '/home/xzy/datasets/MARS-v160809/bbox_test/0010'
    clips_path1 = '/home/xzy/datasets/MARS-v160809/bbox_test/0050'
    clip0 = []
    clip1 = []
    for i in range(1, 16):
        img0_path = os.path.join(clips_path0, f'0010C1T0001F{i:0>3}.jpg')
        img1_path = os.path.join(clips_path1, f'0050C2T0001F{i:0>3}.jpg')
        clip0.append(cv2.imread(img0_path))
        clip1.append(cv2.imread(img1_path))
    clips = [clip0, clip1]
    t0 = time.time()
    person_ReID_video =  hypergraph_reid_()
    features = person_ReID_video.run(clips)
    t1 = time.time()
    print(features.size())
    print(features)
    print(t1-t0)
    print(torch.mm(features, features.t()))
