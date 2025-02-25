from flask import Flask, render_template, request, jsonify, session
from flask_sqlalchemy import SQLAlchemy
import os
import random
import uuid
from dotenv import load_dotenv
from app.db import db, Conversa, MensagemX, MensagemY, Avaliacao, Proficiencia
from app.models import modelo_x_response, modelo_y_response  # Funções importadas corretamente
import pandas as pd
from app.stats import calculate_statistics, FALLBACK_MSG

import logging
logging.basicConfig(level=logging.ERROR)

# Carregar variáveis de ambiente
load_dotenv()

app = Flask(__name__)

# Configuração do banco de dados SQLite
db_path = os.path.abspath(os.path.join(os.path.dirname(__file__), 'chat.db'))
app.config['SQLALCHEMY_DATABASE_URI'] = f'sqlite:///{db_path}'
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False
app.config['SECRET_KEY'] = os.environ.get('SECRET_KEY', 'sua_chave_secreta_aqui')
db.init_app(app)

# Criar as tabelas no banco de dados
with app.app_context():
    db.create_all()

@app.route('/')
def index():
    session_id = str(uuid.uuid4())
    session['conversa_id'] = session_id
    session['chat_a'] = random.choice(['X', 'Y'])
    session['chat_b'] = 'Y' if session['chat_a'] == 'X' else 'X'
    session['historico'] = []  # Inicializa o histórico vazio
    conversa = Conversa(id=session_id, chat_a=session['chat_a'], chat_b=session['chat_b'])
    db.session.add(conversa)
    db.session.commit()
    return render_template('chat.html', session_id=session_id)

# As funções modelo_x_response e modelo_y_response são usadas diretamente de app.models
# Não redefina essas funções aqui!

@app.route('/send_message', methods=['POST'])
def send_message():
    data = request.json
    mensagem = data['message']
    session_id = session['conversa_id']

    # Recuperar históricos para ambos os modelos
    historico_x = MensagemX.query.filter_by(conversa_id=session_id).order_by(MensagemX.id).all()
    historico_y = MensagemY.query.filter_by(conversa_id=session_id).order_by(MensagemY.id).all()

    historico_list_x = [{'remetente': msg.remetente, 'conteudo': msg.conteudo} for msg in historico_x]
    historico_list_y = [{'remetente': msg.remetente, 'conteudo': msg.conteudo} for msg in historico_y]

    # Gerar respostas para ambos os modelos
    resposta_x = modelo_x_response(mensagem, historico_list_x)  # Resposta para Chat A
    resposta_y = modelo_y_response(mensagem, historico_list_y)  # Resposta para Chat B

    # Salvar as mensagens no banco de dados
    msg_user_x = MensagemX(conversa_id=session_id, remetente='user', conteudo=mensagem)
    msg_model_x = MensagemX(conversa_id=session_id, remetente='model', conteudo=resposta_x)
    msg_user_y = MensagemY(conversa_id=session_id, remetente='user', conteudo=mensagem)
    msg_model_y = MensagemY(conversa_id=session_id, remetente='model', conteudo=resposta_y)

    db.session.add_all([msg_user_x, msg_model_x, msg_user_y, msg_model_y])
    db.session.commit()

    # Retornar respostas separadas para os chats A e B
    return jsonify({
        'resposta_a': resposta_x,
        'resposta_b': resposta_y
    })


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
        
        # Reinicia a sessão
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
        # Em vez de logar, retorne uma mensagem de erro genérica
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

if __name__ == '__main__':
    app.run(debug=True)