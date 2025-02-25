from flask_sqlalchemy import SQLAlchemy
from datetime import datetime

db = SQLAlchemy()

class Conversa(db.Model):
    __tablename__ = 'conversa'
    id = db.Column(db.String(36), primary_key=True)
    chat_a = db.Column(db.String(1))
    chat_b = db.Column(db.String(1))

class MensagemX(db.Model):
    __tablename__ = 'mensagem_x'
    id = db.Column(db.Integer, primary_key=True)
    conversa_id = db.Column(db.String(36), db.ForeignKey('conversa.id'))
    remetente = db.Column(db.String(50))
    conteudo = db.Column(db.Text)

class MensagemY(db.Model):
    __tablename__ = 'mensagem_y'
    id = db.Column(db.Integer, primary_key=True)
    conversa_id = db.Column(db.String(36), db.ForeignKey('conversa.id'))
    remetente = db.Column(db.String(50))
    conteudo = db.Column(db.Text)

class Avaliacao(db.Model):
    __tablename__ = 'avaliacao'
    id = db.Column(db.Integer, primary_key=True)
    conversa_id = db.Column(db.String(36), db.ForeignKey('conversa.id'))
    modelo_vencedor = db.Column(db.String(50))
    nome = db.Column(db.String(100), nullable=True)
    email = db.Column(db.String(120), nullable=True)
    data_hora = db.Column(db.DateTime, default=datetime.utcnow)  # Nova coluna

class Proficiencia(db.Model):
    __tablename__ = 'proficiencia'
    id = db.Column(db.Integer, primary_key=True)
    conversa_id = db.Column(db.String(36), db.ForeignKey('conversa.id'))
    nivel = db.Column(db.String(50))
