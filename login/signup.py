# Importing necessary packages.
from flask import Blueprint, render_template, request
from passlib.hash import bcrypt
from email_validator import validate_email, EmailNotValidError

# Importing users table and SQLAlchemy object.
from tables.users import Users
from tables.common import db

# Importing functions to access the database.
from functions.users import get_username

# Creating a blueprint for signup.
signups = Blueprint('signup', __name__)

# When a user tries to signup, it is checked whether the username is already in use.
# If not, the user is added to the database.
@signups.route('/signup', methods = ['GET', 'POST'])
def signup():
	if request.method == 'GET':
		return render_template('signup.html')
	if request.method == 'POST':
		username = request.form['username']
		try:
			emailinfo = validate_email(username, check_deliverability = True)
			username = emailinfo.normalized
		except EmailNotValidError as e:
			return str(e)
		pass1 = request.form['pass1']
		pass2 = request.form['pass2']
		name = request.form['name']
		if username in get_username():
			return "Username already exists."
		else:
			if pass1 == pass2:
				user = Users(username = username, password = bcrypt.hash(pass1),\
		 name = name, admin = 0)
				db.session.add(user)
				db.session.commit()
				return "New user successfully added."
			else:
				return "Given passwords do not match."