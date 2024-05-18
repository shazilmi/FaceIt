from facenet_pytorch import MTCNN
import cv2
from math import atan, degrees
from scipy import ndimage
from PIL import Image
import numpy as np


def check_extension(filename):
	allowed = ['png', 'jpg', 'jpeg']
	if '.' in filename and filename.rsplit('.', 1)[1].lower() in allowed:
		return True
	return False

def preprocess(file):
	mtcnn = MTCNN(select_largest=False, post_process=False)
	img = cv2.imread(file)
	j = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
	boxes, probs, landmarks = mtcnn.detect(j, landmarks = True)
	theta = atan((landmarks[0][1][1] - landmarks[0][0][1]) /\
			(landmarks[0][1][0] - landmarks[0][0][0]))
	rotated = ndimage.rotate(j, degrees(theta))
	l = mtcnn(rotated).permute(1, 2, 0).int().numpy()
	return l

def save_processed(image, filepath):
	img = Image.fromarray((image).astype(np.uint8))
	img.save(filepath)