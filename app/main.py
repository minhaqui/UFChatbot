import time
from flask import Flask, render_template, request, jsonify, session
from flask_sqlalchemy import SQLAlchemy
from sqlalchemy import inspect, text
import os
import random
import uuid
from dotenv import load_dotenv
from app.db import db, Conversa, MensagemX, MensagemY, Avaliacao, Proficiencia
from app.models import modelo_x_response, modelo_y_response
import pandas as pd
from app.stats import calculate_statistics, FALLBACK_MSG
# from app import create_app
import sys
from pathlib import Path

# Adicionar o diretório 'app/' ao sys.path
BASE_DIR = Path(__file__).resolve().parent
if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))

import logging

def create_app():
    # Criar a instância do Flask dentro da função
    app = Flask(__name__)

    # Configurar o logging para exibir no console
    logging.basicConfig(level=logging.DEBUG)

    # Adicionar um handler ao logger da aplicação
    app.logger.setLevel(logging.DEBUG)
    handler = logging.StreamHandler()
    handler.setLevel(logging.DEBUG)
    app.logger.addHandler(handler)

    # Carregar variáveis de ambiente
    load_dotenv()

    # Configuração do banco de dados PostgreSQL
    db_uri = os.getenv('DATABASE_URL')
    if not db_uri:
        raise ValueError("DATABASE_URL não está definido nas variáveis de ambiente.")
    app.config['SQLALCHEMY_DATABASE_URI'] = db_uri
    app.config['SQLALCHEMY_ENGINE_OPTIONS'] = {"connect_args": {"options": "-csearch_path=ufchatbot"}}
    app.config['SECRET_KEY'] = os.getenv('SECRET_KEY')
    app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False

    # Inicializar o SQLAlchemy com a aplicação
    db.init_app(app)

    # Criar tabelas no banco de dados, se necessário
    with app.app_context():
        inspector = inspect(db.engine)
        tables = ['conversa', 'mensagem_x', 'mensagem_y', 'avaliacao', 'proficiencia']
        for table in tables:
            if not inspector.has_table(table, schema='ufchatbot'):
                # Create the schema if it doesn’t exist
                with db.engine.connect() as connection:
                    connection.execute(text("CREATE SCHEMA IF NOT EXISTS ufchatbot"))
                db.create_all()
            else:
                print(f"Tabela {table} já existe no esquema ufchatbot.")

    # Definição das rotas
    @app.route('/')
    def index():
        app.logger.debug("Entering index route")
        session_id = str(uuid.uuid4())
        session['conversa_id'] = session_id
        session['chat_a'] = random.choice(['X', 'Y'])
        session['chat_b'] = 'Y' if session['chat_a'] == 'X' else 'X'
        session['historico'] = []
        conversa = Conversa(id=session_id, chat_a=session['chat_a'], chat_b=session['chat_b'])
        db.session.add(conversa)
        db.session.commit()
        app.logger.debug("Exiting index route")
        return render_template('chat.html', session_id=session_id)

    @app.route('/send_message', methods=['POST'])
    def send_message():
        data = request.json
        mensagem = data['message']
        session_id = session['conversa_id']

        # Recuperar histórico com base no modelo associado a cada chat
        if session['chat_a'] == 'X':
            historico_a = MensagemX.query.filter_by(conversa_id=session_id).order_by(MensagemX.id).all()
            historico_b = MensagemY.query.filter_by(conversa_id=session_id).order_by(MensagemY.id).all()
        else:
            historico_a = MensagemY.query.filter_by(conversa_id=session_id).order_by(MensagemY.id).all()
            historico_b = MensagemX.query.filter_by(conversa_id=session_id).order_by(MensagemX.id).all()

        historico_list_a = [{'remetente': msg.remetente, 'conteudo': msg.conteudo} for msg in historico_a]
        historico_list_b = [{'remetente': msg.remetente, 'conteudo': msg.conteudo} for msg in historico_b]

        # Gerar respostas com base no modelo associado a cada chat
        if session['chat_a'] == 'X':
            resposta_a = modelo_x_response(mensagem, historico_list_a)  # Modelo puro
            resposta_b = modelo_y_response(mensagem, historico_list_b)  # Modelo com RAG
        else:
            resposta_a = modelo_y_response(mensagem, historico_list_a)  # Modelo com RAG
            resposta_b = modelo_x_response(mensagem, historico_list_b)  # Modelo puro

        # Salvar mensagens no banco de dados
        msg_user_x = MensagemX(conversa_id=session_id, remetente='user', conteudo=mensagem)
        msg_model_x = MensagemX(conversa_id=session_id, remetente='model', conteudo=resposta_a if session['chat_a'] == 'X' else resposta_b)
        msg_user_y = MensagemY(conversa_id=session_id, remetente='user', conteudo=mensagem)
        msg_model_y = MensagemY(conversa_id=session_id, remetente='model', conteudo=resposta_b if session['chat_a'] == 'X' else resposta_a)

        db.session.add_all([msg_user_x, msg_model_x, msg_user_y, msg_model_y])
        db.session.commit()

        return jsonify({'resposta_a': resposta_a, 'resposta_b': resposta_b})

    @app.route('/evaluate', methods=['POST'])
    def evaluate():
        try:
            data = request.json
            proficiencia = data.get('proficiencia')
            if not proficiencia:
                return jsonify({'error': 'O nível de proficiência é obrigatório.'}), 400

            session_id = session.get('conversa_id')
            winner = data.get('winner')
            nome = data.get('nome')
            email = data.get('email')
            modelo_vencedor = 'Chat A' if winner == 'Chat A' else 'Chat B'

            avaliacao = Avaliacao(conversa_id=session_id, modelo_vencedor=modelo_vencedor, nome=nome, email=email)
            proficiencia_entry = Proficiencia(conversa_id=session_id, nivel=proficiencia)
            db.session.add_all([avaliacao, proficiencia_entry])
            db.session.commit()

            session.pop('historico', None)
            new_session_id = str(uuid.uuid4())
            session['conversa_id'] = new_session_id
            session['chat_a'] = random.choice(['X', 'Y'])
            session['chat_b'] = 'Y' if session['chat_a'] == 'X' else 'X'
            conversa = Conversa(id=new_session_id, chat_a=session['chat_a'], chat_b=session['chat_b'])
            db.session.add(conversa)
            db.session.commit()

            return jsonify({'status': 'Avaliação registrada', 'winner': modelo_vencedor})
        except Exception as e:
            return jsonify({'error': 'Erro ao processar a avaliação. Por favor, tente novamente.'}), 500

    @app.route('/reset', methods=['POST'])
    def reset():
        session_id = session['conversa_id']
        MensagemX.query.filter_by(conversa_id=session_id).delete()
        MensagemY.query.filter_by(conversa_id=session_id).delete()
        db.session.commit()
        session.pop('historico', None)
        session['conversa_id'] = str(uuid.uuid4())
        session['chat_a'] = random.choice(['X', 'Y'])
        session['chat_b'] = 'Y' if session['chat_a'] == 'X' else 'X'
        conversa = Conversa(id=session['conversa_id'], chat_a=session['chat_a'], chat_b=session['chat_b'])
        db.session.add(conversa)
        db.session.commit()
        return jsonify({'status': 'Conversa resetada'})

    @app.route('/resultados')
    def resultados():
        avaliacoes_df = pd.read_sql(db.session.query(Avaliacao).statement, db.engine)
        conversas_df = pd.read_sql(db.session.query(Conversa).statement, db.engine)
        proficiencias_df = pd.read_sql(db.session.query(Proficiencia).statement, db.engine)
        stats = calculate_statistics(avaliacoes_df, conversas_df, proficiencias_df)
        return render_template('resultados.html', 
                               desc_stats=stats.get('desc_stats', FALLBACK_MSG),
                               tabela_avaliacoes=stats.get('tabela_avaliacoes', FALLBACK_MSG),
                               teste_hipotese=stats.get('teste_hipotese', FALLBACK_MSG))

    @app.route('/sobre')
    def sobre():
        return render_template('sobre.html')

    # Retornar a aplicação configurada
    return app

# Criar a instância da aplicação
app = create_app()

if __name__ == '__main__':
    app.run(debug=True)