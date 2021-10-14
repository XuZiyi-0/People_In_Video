import cv2
import matplotlib.pyplot as plt
import time
import os, sys
sys.path.insert(0, os.path.abspath('../../'))

from modules.data_reader.general import VideoReader
from modules.det_and_trk.det_and_trk import Detection_Tracking

def compute_color_for_id(label):
	"""
	根据踪片id选择一个颜色
	"""
	palette = (2**11-1, 2**15-1, 2**20-1)
	color = [int((p * (label ** 2 - label + 1)) % 255) for p in palette]
	return tuple(color)

def plot_one_bbox(bbox, frame, color, label, line_thickness=None):
	tl = line_thickness or round(0.002 * (frame.shape[0]+frame.shape[1])/2) + 1
	c1, c2 = (int(bbox[0]), int(bbox[1])), (int(bbox[2]), int(bbox[3]))
	cv2.rectangle(frame, c1, c2, color, thickness=tl, lineType=cv2.LINE_AA)
	if label:
		tf = max(tl-1, 1) # font thickness
		t_size = cv2.getTextSize(label, 0, fontScale=tl/3, thickness=tf)[0]
		c2 = (c1[0] + t_size[0], c1[1] - t_size[1] -3)
		cv2.rectangle(frame, c1, c2, color, -1, cv2.LINE_AA)
		cv2.putText(frame, label, (c1[0], c1[1] -2), 0, tl/3, [255, 255, 255], thickness=tf, lineType=cv2.LINE_AA)


def demo(data_root, video_name, interval=10):
	video_path = os.path.join(data_root, 'videos', video_name)
	video_save_path = os.path.join(data_root, 'results_tracklets', video_name.split('.')[0], video_name)
	frame_reader = VideoReader()
	frame_reader.capture(video_path)

	det_and_trk = Detection_Tracking()

	ret, frame = frame_reader.read()
	fps = frame_reader.video.get(cv2.CAP_PROP_FPS)
	w, h = frame.shape[1], frame.shape[0]
	vid_writer = cv2.VideoWriter(video_save_path, cv2.VideoWriter_fourcc(*'mp4v'), fps, (w, h))

	frame_idx = 0
	t0 = time.time()
	while ret:
		tracklets, infos = det_and_trk.run(frame)
		print(frame_idx)
		# print(tracklets)
		# print(infos)
		# print(infos.keys())
		# print('#'*64)
		if len(tracklets) > 0:
			# 保存踪片图像，每10帧保存一次
			if frame_idx%interval == 0:
				tracklets_folder = os.path.join(data_root, 'results_tracklets', video_name.split('.')[0], 'tracklets')
				if not os.path.exists(tracklets_folder):
					os.mkdir(tracklets_folder)
				for j, (tracklet, det_score, associate_metric, distance) in enumerate(zip(tracklets, infos['det_score'], infos['associate_metric'], infos['distance'])):
					id = tracklet[4]
					save_folder = os.path.join(data_root, 'results_tracklets', video_name.split('.')[0], 'tracklets',str(id))
					if not os.path.exists(save_folder):
						os.mkdir(save_folder)
					label = f'id={id},' \
					        f'det_score={det_score:.2f},' \
					        f'{associate_metric}={distance:.4f}'
					img = frame[tracklet[1]:tracklet[3], tracklet[0]:tracklet[2], :]
					img_rbg = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
					plt.imsave(os.path.join(save_folder, str(frame_idx)+label+'.jpg'), img_rbg)
			# 保存到输出视频
			for j, (tracklet, det_score, associate_metric, distance) in enumerate(zip(tracklets, infos['det_score'], infos['associate_metric'], infos['distance'])):
				bbox = tracklet[0:4]
				id = tracklet[4]
				label = f'id:{id}|{distance:.4f}'
				color = compute_color_for_id(id)
				plot_one_bbox(bbox, frame, label=label, color=color, line_thickness=2)

		vid_writer.write(frame)
		frame_idx += 1
		ret, frame = frame_reader.read()
	t1 = time.time()
	print("%.3fs/frame"%((t1-t0)/frame_idx))

if __name__=='__main__':
	demo(data_root='/home/xzy/projects/People_In_Video/test_data/hurenVSyongshi', video_name='clip0.mp4', interval=1)