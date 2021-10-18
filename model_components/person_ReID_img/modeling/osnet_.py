import os
import torch
import cv2
import numpy as np
import time

from model_components.person_ReID_img.modeling.osnet.osnet import osnet_x1_0

from torchvision import transforms
from PIL import Image
from model_components.configs.person_ReID_img import osnet_cfg

from model_components.person_ReID_img.modeling.build import PERSON_REID_IMG_REGISTRY


@PERSON_REID_IMG_REGISTRY.register()
def build_osnet():
    return osnet_()


class osnet_:
    def __init__(self, cfg=osnet_cfg.clone()):
        self.cfg = cfg
        self.cfg.freeze()
        self.device = torch.device('cuda:%d' % self.cfg.DEVICE_ID[0])

        self.model = osnet_x1_0(num_classes=self.cfg.NUM_CLASS, pretrain=False, loss=self.cfg.Loss)
        self.model.to(self.device)
        self.model = torch.nn.DataParallel(self.model, device_ids=cfg.DEVICE_ID)
        self.model.eval()
        with torch.no_grad():
            self.model.load_state_dict(torch.load(self.cfg.CHECKPOINT, map_location=self.device)['state_dict'])

        self.transformer = transforms.Compose([
            transforms.Resize(self.cfg.IMG_SIZE, interpolation=3),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])

    # def fliplr(self, img):
    #     inv_idx = torch.arange(img.size(3) - 1, -1, -1).long()  # N x c x H x W
    #     img_flip = img.index_select(3, inv_idx)
    #     return img_flip

    def preprocess(self, imgs0):
        imgs = []
        for i in range(len(imgs0)):
            img = cv2.cvtColor(imgs0[i], cv2.COLOR_BGR2RGB)
            img = Image.fromarray(img)
            img = self.transformer(img)
            img = img.unsqueeze(0)
            imgs.append(img)
        imgs = torch.cat(imgs)
        return imgs

    def run(self, imgs):
        with torch.no_grad():
            imgs = self.preprocess(imgs)
            features = self.model.forward(imgs)
        return features


if __name__ == '__main__':
    # 单元测试
    person_ReID_img = osnet_()
    imgs = []
    imgs_path = '/home/xzy/projects/People_In_Video/test_data/person_ReID_img/query'
    img_names = os.listdir(imgs_path)
    for img_name in img_names:
        imgs.append(cv2.imread(os.path.join(imgs_path, img_name)))

    for i in range(10):
        imgs_ = imgs[:]
        t0 = time.time()
        person_ReID_img.run(imgs_)
        t1 = time.time()
        print(t1 - t0)
