import os
# Workaround para o erro de OpenMP: garante que apenas uma runtime seja usada.
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import unittest
import json
from app.main import app, db, Conversa, MensagemX, MensagemY, Avaliacao, Proficiencia

class TestRoutes(unittest.TestCase):
    def setUp(self):
        app.config['TESTING'] = True
        # Use um banco de dados em memória para os testes (opcional, se preferir isolar)
        app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///:memory:'
        self.client = app.test_client()
        with app.app_context():
            db.create_all()

    def tearDown(self):
        with app.app_context():
            db.session.remove()
            db.drop_all()

    def test_index_route(self):
        response = self.client.get('/')
        self.assertEqual(response.status_code, 200)
        self.assertIn(b'Chat Comparativo', response.data)

    def test_send_message(self):
        # Acessa a página inicial para configurar a sessão
        self.client.get('/')
        data = {'message': 'Teste de mensagem'}
        response = self.client.post('/send_message',
                                    data=json.dumps(data),
                                    content_type='application/json')
        self.assertEqual(response.status_code, 200)
        json_data = json.loads(response.data)
        self.assertIn('resposta_a', json_data)
        self.assertIn('resposta_b', json_data)

    def test_evaluate(self):
        # Configura a sessão inicial
        self.client.get('/')
        data = {
            'winner': 'Chat A',
            'nome': 'Teste',
            'email': 'teste@teste.com',
            'proficiencia': 'Intermediario'
        }
        response = self.client.post('/evaluate',
                                    data=json.dumps(data),
                                    content_type='application/json')
        self.assertEqual(response.status_code, 200)
        json_data = json.loads(response.data)
        self.assertIn('status', json_data)
        self.assertEqual(json_data['status'], 'Avaliação registrada')

    def test_reset(self):
        self.client.get('/')
        # Envia uma mensagem para criar histórico
        data = {'message': 'Mensagem para reset'}
        self.client.post('/send_message',
                         data=json.dumps(data),
                         content_type='application/json')
        # Chama a rota de reset
        response = self.client.post('/reset')
        self.assertEqual(response.status_code, 200)
        json_data = json.loads(response.data)
        self.assertIn('status', json_data)
        self.assertEqual(json_data['status'], 'Conversa resetada')

    def test_resultados(self):
        # Simula uma avaliação para garantir que haja dados a serem exibidos
        with app.app_context():
            conversa = Conversa(id='test123', chat_a='X', chat_b='Y')
            db.session.add(conversa)
            db.session.commit()
            avaliacao = Avaliacao(conversa_id='test123', modelo_vencedor='Chat A', nome='Teste', email='teste@teste.com')
            db.session.add(avaliacao)
            proficiencia = Proficiencia(conversa_id='test123', nivel='Intermediario')
            db.session.add(proficiencia)
            db.session.commit()

        response = self.client.get('/resultados')
        self.assertEqual(response.status_code, 200)
        self.assertIn(b'Resultados Preliminares', response.data)

if __name__ == '__main__':
    unittest.main()
