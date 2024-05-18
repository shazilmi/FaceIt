from model.model import Net
from torchvision.transforms.functional import convert_image_dtype, to_tensor
import torchvision.transforms.functional as F
import torch
import numpy as np
from torch import nn
from skimage.segmentation import mark_boundaries
import matplotlib.pyplot as plt
import lime
from lime import lime_image
import os

model = Net()
checkpoint = torch.load('model/model_new.pt')
model.load_state_dict(checkpoint)
model.eval()

def get_prediction(img):
	img = to_tensor(img)
	image = convert_image_dtype(img)
	sm = nn.Softmax(dim = 1)
	output1 = []
	output2 = []
	for i in range(10):
		output = model(image.unsqueeze(0))[0]
		outputs = sm(output)
		output1.append(outputs[0][0])
		output2.append(outputs[0][1])
	sum1, sum2 = 0, 0
	for i in output1:
		sum1 += i
	for j in output2:
		sum2 += j
	if sum1 > sum2:
		highest_var_label = 1
		prob = float(sum1 / 10)
	else:
		highest_var_label = 0
		prob = round(float(sum2 / 10), 3)
		if prob < 0.82:
			highest_var_label = 1
	return highest_var_label, prob

device = 'cpu'

def single_predict(image):
  model.eval()
  model.to(device)
  logits = model(image.unsqueeze(0))
  probs = F.softmax(logits[0], dim = 1)
  return probs.detach().cpu().numpy()


def thelime(image, filename, dir):
	explainer = lime_image.LimeImageExplainer()
	explanation = explainer.explain_instance(np.array(image.permute(1, 2, 0)),
											single_predict,
											top_labels=[0,1],
											hide_color=0,
											num_samples=1)

	temp, mask = explanation.get_image_and_mask(explanation.top_labels[0], positive_only=True, num_features= 10, hide_rest=False)
	img_boundry1 = mark_boundaries(temp, mask)
	

	temp, mask = explanation.get_image_and_mask(explanation.top_labels[1], positive_only=True, num_features=10, hide_rest=False)
	img_boundry2 = mark_boundaries(temp, mask)
	fig, axs = plt.subplots(1, 2)
	axs[0] = img_boundry1
	axs[1] = img_boundry2
	plt.savefig(os.path.join(dir, 'lime', filename))