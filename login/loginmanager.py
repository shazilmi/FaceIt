# Importing necessary packages.
from flask_login import LoginManager
from flask import redirect

# Importing necessary functions to access the database.
from functions.users import get_username
from functions.users import get_admin_name

# Importing SQLAlchemy object.
from tables.common import db

# Importing User class for login.
from login.usermixin import User

# Creating LoginManager object.
lm = LoginManager()

# Setting up how to load a user from a request and from its session.
@lm.user_loader
def user_loader(username):
	usernames = get_username()
	if username not in usernames:
		return None
	user = User()
	user.id = username
	admin, name = get_admin_name(username)
	user.admin = admin
	user.name = name
	return user

@lm.request_loader
def request_loader(request):
	usernames = get_username()
	username = request.form.get('username')
	if username not in usernames:
		return None
	user = User()
	user.id = username
	admin, name = get_admin_name(username)
	user.admin = admin
	user.name = name
	return user

# Handling unauthorized requests.
@lm.unauthorized_handler
def kick():
	return redirect('login')