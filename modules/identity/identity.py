import cv2
import torch
import numpy as np

import sys, os
sys.path.insert(0, os.path.abspath('../../'))

from modules.data_reader import PersonReIDImgSampleReader
from model_components import PersonReIDImg

class PersonReIDImg_FeatureComparison:
	def __init__(self, sample_path):
		self.sample_reader = PersonReIDImgSampleReader(sample_path)
		self.sample_reader.read_features()
		self.samples = self.sample_reader.samples
		self.person_reid_img = PersonReIDImg()

	def run(self, imgs):

		qf = self.person_reid_img.run(imgs)
		gf = torch.transpose(self.samples['feature'], 0, 1)
		score = torch.mm(qf, gf).cpu().numpy()

		index = np.argsort(score) # from small to large
		index = index[:, ::-1]

		res = {}
		res['score'] = []
		res['person_name'] = []
		res['img_name'] = []
		for i in range(index.shape[0]):
			res['score'].append([score[i, idx] for idx in index[i]])
			res['person_name'].append([self.samples['person_name'][idx] for idx in index[i]])
			res['img_name'].append([self.samples['img_name'][idx] for idx in index[i]])
		return res





if __name__ == '__main__':
	sample_path = '/home/xzy/projects/Worksite_Monitors/test_data/2021湖人vs勇士/people_samples'
	test_img_path = '/home/xzy/projects/Worksite_Monitors/test_data/2021湖人vs勇士/results_tracklets/clip0/tracklets/18'
	img_names = os.listdir(test_img_path)[2:10]
	test_imgs = []
	for img_name in img_names:
		img = cv2.imread(os.path.join(test_img_path, img_name))
		test_imgs.append(img)

	identity_module = PersonReIDImg_FeatureComparison(sample_path)
	res = identity_module.run(test_imgs)
	for i in range(8):
		print(img_names[i])
		print(res['score'][i])
		print(res['person_name'][i])
		print(res['img_name'][i])
		print('#'*64)