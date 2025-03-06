from sqlalchemy import create_engine, text
from main import app


with app.app_context():
    engine = create_engine(app.config['SQLALCHEMY_DATABASE_URI'])
    with engine.connect() as connection:
        connection.execute(text("CREATE SCHEMA IF NOT EXISTS ufchatbot"))
        connection.commit()