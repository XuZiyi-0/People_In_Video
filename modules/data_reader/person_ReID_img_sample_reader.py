import cv2
import torch
import tqdm, math

import sys, os
sys.path.insert(0, '../../')

from model_components.person_ReID_img import PersonReIDImg

class PersonReIDImgSampleReader:
	def __init__(self, sample_path):
		self.sample_path = sample_path
		self.samples = {}
		self.samples['img'] = []
		self.samples['img_name'] = []
		self.samples['person_name'] = []
		person_names = os.listdir(sample_path)
		for person_name in person_names:
			if os.path.isfile(os.path.join(sample_path, person_name)):
				continue
			img_names = os.listdir(os.path.join(sample_path, person_name))
			for img_name in img_names:
				img = cv2.imread(os.path.join(sample_path, person_name, img_name))
				self.samples['img'].append(img)
				self.samples['img_name'].append(img_name)
				self.samples['person_name'].append(person_name)
	
	def save_features(self, batch_size=32):
		person_ReID_img = PersonReIDImg()
		features = torch.FloatTensor()
		n = len(self.samples['img'])
		p = 0
		print('extracting sample features')
		for i in tqdm.tqdm(range(0, math.ceil(n/batch_size))):
			idx0 = batch_size*p
			idx1 = batch_size*(p+1)
			idx1 = idx1 if idx1<n else n
			f = person_ReID_img.run(self.samples['img'][idx0 : idx1])
			features = torch.cat((features, f), 0)
			p+=1
		save_path = os.path.join(self.sample_path,'features.pt')
		torch.save(features, save_path)
		print('features saved in', save_path)

	def read_features(self):
		self.samples['feature'] = torch.load(os.path.join(self.sample_path, 'features.pt'))
		assert len(self.samples['img']) == self.samples['feature'].size()[0]

if __name__ == '__main__':
	sample_path = '/home/xzy/projects/Worksite_Monitors/test_data/2021湖人vs勇士/people_samples'
	person_ReID_img_sample_reader = PersonReIDImgSampleReader(sample_path)
	person_ReID_img_sample_reader.read_features()