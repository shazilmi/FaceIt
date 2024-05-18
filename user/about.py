from flask import Blueprint, render_template, request, redirect
from flask_login import current_user, login_required

# Creating a blueprint for the user dashboard.
abouts = Blueprint('about', __name__)
@abouts.route('/about', methods = ['GET'])
def about():
	if request.method == 'GET':
		return render_template('about.html')