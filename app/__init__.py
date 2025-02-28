
'''
from flask import Flask
from flask_sqlalchemy import SQLAlchemy

db = SQLAlchemy()

def create_app():
    app = Flask(__name__)
    # Config and routes as in main.py
    db.init_app(app)
    return app

    '''