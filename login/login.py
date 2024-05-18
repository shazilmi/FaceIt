from flask import Blueprint, render_template, request, redirect
from passlib.hash import bcrypt
from flask_login import login_user
from login.usermixin import User

# Importing functions to access the database.
from functions.users import get_password, get_name, get_username, get_admin

# Creating a blueprint for login.
logins = Blueprint('login', __name__)

# When a user tries to login, it is checked whether the username exists in the database.
# If such a username exists, and the password associated with the username is correctly entered,
# then the user is logged in.
@logins.route('/login', methods = ['GET', 'POST'])
def login():
	if request.method == 'GET':
		return render_template('login.html')
	if request.method == 'POST':
		username = request.form['username']
		password = request.form['pass']
		if username in get_username():
			user_pass = get_password(username)
			if bcrypt.verify(password, user_pass):
				user = User()
				user.id = username
				user.name = get_name(username)
				user.admin = get_admin(username)
				try:
					if request.form['remember'] == 'on':
						login_user(user, remember = True)
				except:
					login_user(user)
				if user.admin == 1:
					return redirect('admindash')
				else:
					return redirect('userdash')
			else:
				return "Incorrect credentials used."
		else:
			return "Incorrect credentials used."