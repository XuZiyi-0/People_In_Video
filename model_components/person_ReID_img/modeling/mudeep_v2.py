import os
import torch
import cv2
import numpy as np
import time

from model_components.person_ReID_img.modeling.MuDeep_v2.network import MuDeep_v2
from model_components.utils import BGR_hwc_to_RGB_chw
from model_components.configs.person_ReID_img import mudeepv2_cfg

from model_components.person_ReID_img.modeling.build import PERSON_REID_IMG_REGISTRY

@PERSON_REID_IMG_REGISTRY.register()
def build_mudeepv2():
	return MuDeep_v2_()

class MuDeep_v2_:
	def __init__(self, cfg=mudeepv2_cfg.clone()):
		self.cfg = cfg
		self.cfg.freeze()
		self.device = torch.device('cuda:%d'%self.cfg.DEVICE_ID[0])

		self.model = MuDeep_v2(num_class=self.cfg.NUM_CLASS, num_scale=3, pretrain=True)
		self.model.to(self.device)
		self.model = torch.nn.DataParallel(self.model, device_ids=cfg.DEVICE_ID)
		self.model.eval()
		with torch.no_grad():
			self.model.load_state_dict(torch.load(self.cfg.CHECKPOINT, map_location=self.device)['state_dict'])

	def fliplr(self, img):
		inv_idx = torch.arange(img.size(3) - 1, -1, -1).long()  # N x c x H x W
		img_flip = img.index_select(3, inv_idx)
		return img_flip

	def preprocess(self, imgs0):
		imgs = []
		for i in range(len(imgs0)):
			img = cv2.resize(imgs0[i], dsize=(self.cfg.IMG_SIZE[1], self.cfg.IMG_SIZE[0]), interpolation=cv2.INTER_CUBIC)
			img = BGR_hwc_to_RGB_chw(img)
			imgs.append(img)
		imgs = np.array(imgs)
		imgs = torch.from_numpy(imgs).float()
		return imgs


	def run(self, imgs):
		with torch.no_grad():
			imgs = self.preprocess(imgs)
			features = torch.FloatTensor()
			n, c, h, w = imgs.size()
			ff = torch.FloatTensor(n, 512 * 12).zero_()
			for i in range(2):
				if (i == 1):
					imgs = self.fliplr(imgs)
				input_img = imgs.to(self.device)
				out_1_g_0, out_1_l_0, out_1_l_1, out_1_l_2, \
				out_2_g_0, out_2_l_0, out_2_l_1, out_2_l_2, \
				out_3_g_0, out_3_l_0, out_3_l_1, out_3_l_2 = self.model(input_img, test=True)
				outputs = torch.cat((out_1_g_0, out_1_l_0, out_1_l_1, out_1_l_2,
				                     out_2_g_0, out_2_l_0, out_2_l_1, out_2_l_2,
				                     out_3_g_0, out_3_l_0, out_3_l_1, out_3_l_2), dim=1)
				f = outputs.data.cpu()
				ff = ff + f / 2.
			fnorm = torch.norm(ff, p=2, dim=1, keepdim=True)
			ff = ff.div(fnorm.expand_as(ff))
			features = torch.cat((features, ff), 0)
		return features

if __name__ == '__main__':
	# 单元测试
	person_ReID_img = MuDeep_v2_()
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
		print(t1-t0)