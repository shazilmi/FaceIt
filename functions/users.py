# Importing the SQLAlchemy object.
from tables.common import db

# Importing users table.
from tables.users import Users

# Function to return usernames stored in the table.
def get_username():
	theusers = db.session.execute(db.select(Users.username)).all()
	users = []
	for i in theusers:
		users.append(i[0])
	return users

# Function to return password of given username.
def get_password(username):
	thepass = db.session.execute(db.select(Users.password).filter_by(\
		username = username)).first()
	return thepass[0]

# Function to return usernames and passwords stored in the database.
def get_username_password():
	users_passes = db.session.execute(db.select(Users.username, Users.password)).all()
	user_pass = {}
	for i in users_passes:
		user_pass[i[0]] = i[1]
	return user_pass

# Function to return usernames, admins and names stored in the database.
def get_username_admin_name():
	users_admins_names = db.session.execute(db.select(Users.username,\
						    Users.admin, Users.name)).all()
	user_admin_name = []
	for i in users_admins_names:
		appendlist = []
		appendlist.append(i[0])
		appendlist.append(i[1])
		appendlist.append(i[2])
		user_admin_name.append(appendlist)
	return user_admin_name

# Function to check whether a given user is  an admin and to return the name of the user.
def get_admin_name(username):
	admin_name = db.session.execute(db.select(Users.admin, Users.name).filter_by(\
		username = username)).first()
	admin = admin_name[0]
	name = admin_name[1]
	return admin, name

# Function to get name given username.
def get_name(username):
	thename = db.session.execute(db.select(Users.name).filter_by(\
		username = username)).first()
	return thename[0]

# Function to check whether a given user is an admin.
def get_admin(username):
	theadmin = db.session.execute(db.select(Users.admin).filter_by(\
		username = username)).first()
	return theadmin[0]