from tables.common import db

# User table contains username, name and password.
class Users(db.Model):
	username = db.Column(db.String, primary_key = True)
	name = db.Column(db.String)
	password = db.Column(db.String)
	admin = db.Column(db.Integer)