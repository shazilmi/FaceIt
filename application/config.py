# Importing necessary packages.
import os

# Specifying the configuration for the app.
class AppConfig():
	SQLALCHEMY_DATABASE_URI = "sqlite:///../database/faceit.db/"
	DEBUG = True
	SECRET_KEY = '82tmq*20mzo0!'
	MAX_CONTENT_LENGTH = 50 * 1024 * 1024
	UPLOAD_FOLDER = '/static/uploads'