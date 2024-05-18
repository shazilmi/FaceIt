from flask import Blueprint, render_template, request, redirect
from flask_login import current_user, login_required

# Importing functions to access the database.

# Creating a blueprint for the admin dashboard.
admindashs = Blueprint('admindash', __name__)
@admindashs.route('/admindash', methods = ['GET'])
@login_required
def dashboard():
	try:
		admin = current_user.admin
		if admin == 0:
			return "You're not an admin."
		if request.method == 'GET':
			name = current_user.name
			return render_template('admindash.html', name = name)
	except:
		return redirect('login')