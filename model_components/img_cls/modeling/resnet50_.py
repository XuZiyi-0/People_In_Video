import torch
import os
import pandas as pd
import cv2
import os, sys
import torchvision.transforms as transforms
sys.path.insert(0, os.path.abspath('../../img_cls/modeling'))

from model_components.configs.img_cls import resnet50_cfg
from model_components.img_cls.modeling.resnet50.net import ResNet50
from model_components.img_cls.modeling.build import IMG_CLS_REGISTRY

@IMG_CLS_REGISTRY.register()
def build_resnet50():
	return Resnet50()

class Resnet50:
	def __init__(self, cfg=resnet50_cfg.clone()):
		self.cfg = cfg
		self.cfg.freeze()
		#self.img_size = cfg.IMG_SIZE
		self.device = torch.device('cuda:%d'%self.cfg.DEVICE_ID)
		self.model = ResNet50()
		self.model = torch.nn.DataParallel(self.model, device_ids=(0,))
		checkpoint = torch.load(self.cfg.CHECKPOINT)
		self.model.load_state_dict(checkpoint)
		#self.model.half()  # to FP16

	def run(self, img):
		img = cv2.resize(img, (256, 256))

		tf = transforms.ToTensor()
		img= tf(img)
		img=img.unsqueeze(0)
		# img.requires_grad_(False)
		with torch.no_grad():
			pred = self.model(img).cpu()
		# pred.detach_()
		return pred

if __name__ == '__main__':
	img_clser = build_resnet50(resnet50_cfg.clone())
	img = cv2.imread(os.path.abspath('home/gyj/resnet/data/cat.0.jpg'))
	pred = img_clser.run(img)
	print(pred)