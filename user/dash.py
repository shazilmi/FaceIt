from flask import Blueprint, render_template, request, redirect
from flask_login import current_user, login_required

# Creating a blueprint for the user dashboard.
userdashs = Blueprint('userdash', __name__)
@userdashs.route('/userdash', methods = ['GET'])
@login_required
def dashboard():
	try:
		if request.method == 'GET':
			name = current_user.name
			return render_template('userdash.html', name = name)
	except:
		return redirect('login')