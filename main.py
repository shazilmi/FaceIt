# Importing necessary packages.
from flask import Flask, redirect, render_template

# Importing config
from application.config import AppConfig

# Creating Flask app and configuring.
app = Flask(__name__)
app.config.from_object(AppConfig)

# Importing the LoginManager instance.
from login.loginmanager import lm
lm.init_app(app)

# Importing the SQLAlchemy instance.
from tables.common import db
db.init_app(app)

# Function to create databases and initial admin.
from application.create_db import create_db
with app.app_context():
	create_db()

# Adding blueprints
from application.blueprints import add_blueprints
add_blueprints(app)

# Redirect to admin dashboard page for now.
@app.route('/', methods = ['GET'])
def home():
	return render_template('index.html')

# Running the app.
app.run()