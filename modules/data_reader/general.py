import cv2

class VideoReader:
	def __init__(self):
		self.video = None

	def capture(self, path):
		self.video = cv2.VideoCapture(path)

	def read(self):
		ret, frame = self.video.read()
		return ret, frame