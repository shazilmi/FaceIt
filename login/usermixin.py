# Importing necessary packages.
from flask_login import UserMixin

# Class User derived from UserMixin.
class User(UserMixin):
	id = None
	name = None
	admin = None