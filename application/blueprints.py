# Importing the Blueprints to be registered.
from login.login import logins
from login.signup import signups
from admin.dash import admindashs
from user.dash import userdashs
from user.taketest import take_tests
from user.result import results
from admin.lime import limes
from user.about import abouts

# Function to register all imported blueprints
def add_blueprints(app):
	app.register_blueprint(logins)
	app.register_blueprint(signups)
	app.register_blueprint(admindashs)
	app.register_blueprint(userdashs)
	app.register_blueprint(take_tests)
	app.register_blueprint(results)
	app.register_blueprint(limes)
	app.register_blueprint(abouts)