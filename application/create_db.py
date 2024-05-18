import os

# Importing SQLAlchemy object.
from tables.common import db

# Importing tables.
from tables.users import Users

# Importing function to list currently stored usernames.
from functions.users import get_username

# Importing hash function
from passlib.hash import bcrypt

# Create the tables in the database.
# If there are no users present, create an admin user by default.
# Name, username and password are set from environment variables.
def create_db():
	db.create_all()
	if get_username() == []:
		try:
			name = os.environ['admin_name']
			username = os.environ['admin_username']
			password = bcrypt.hash(os.environ['admin_pass'])
		except:
			name = 'Theadmin'
			username = 'shazil1538@gmail.com'
			password = bcrypt.hash('adminaccess')
		theadmin = Users(name = name, username = username, password = password, admin = 1)
		db.session.add(theadmin)
		db.session.commit()