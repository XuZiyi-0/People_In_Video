import numpy as np

def BGR_hwc_to_RGB_chw(imgs):
	imgs = imgs[:, :, ::-1].transpose(2, 0, 1)
	imgs = np.ascontiguousarray(imgs)
	return imgs
