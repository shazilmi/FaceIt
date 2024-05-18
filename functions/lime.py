from model.model import Net
from torchvision.transforms.functional import convert_image_dtype, to_tensor
import torchvision.transforms.functional as F
import torch
import numpy as np
from torch import nn
from torchvision import transforms
from skimage.segmentation import mark_boundaries
import matplotlib.pyplot as plt
import lime
from lime import lime_image
import os

model = Net()
checkpoint = torch.load('model/model_new.pt')
model.load_state_dict(checkpoint)
model.eval()



device = 'cpu'

def single_predict(image):
	model.eval()
	model.to(device)
	if str(type(image)) == "<class 'numpy.ndarray'>":
		image = torch.from_numpy(image).permute(0, 3, 1, 2)
	if image.shape == torch.Size([3, 160, 160]):
		image = image.unsqueeze(0)
	image = F.convert_image_dtype(image, dtype = torch.float32)
	logits = model(image)
	probs = nn.functional.softmax(logits[0], dim = 1)
	return probs.detach().cpu().numpy()

def thelime(image, filename, dir):
	explainer = lime_image.LimeImageExplainer()
	explanation = explainer.explain_instance(np.array(image.permute(1, 2, 0)),
										  single_predict, top_labels = 2,
										  hide_color = 0, num_samples = 1)
	temp, mask = explanation.get_image_and_mask(explanation.top_labels[0], positive_only=False, num_features= 5, hide_rest=False)
	img_boundry1 = mark_boundaries(temp, mask)
	plt.imshow(img_boundry1)
	thename = 'lime' + filename.split('/')[-1]
	plt.savefig(os.path.join(dir, thename))
	return thename
     

def get_images():
	images = []
	for i in os.listdir('static/uploads'):
		if i.split('.')[-1] == 'jpg' or i.split('.')[-1] == 'JPG':
			images.append(os.path.join('static/uploads', i))
	return images
