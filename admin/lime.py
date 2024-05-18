from flask import Blueprint, render_template, request, redirect
from flask_login import current_user, login_required
from functions.lime import get_images, thelime
from torchvision.io import read_file
from torchvision import transforms
import torch
from PIL import Image
import os

# Importing functions to access the database.

# Creating a blueprint for the admin dashboard.
limes = Blueprint('lime', __name__)
@limes.route('/lime', methods = ['GET', 'POST'])
@login_required
def dashboard():
	if request.method == 'GET':
		images = get_images()
		return render_template('lime.html', imgs = images)
	if request.method == 'POST':
		image = request.form.get('theimage')
		img = Image.open(image)
		transform = transforms.Compose([transforms.PILToTensor()])
		img = transform(img)
		filename = os.path.join('static/uploads/lime', thelime(img, image, 'static/uploads/lime'))
		return render_template('resultdisplay.html', image = filename)