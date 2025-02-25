import unittest
import json
from app.app import app, db, Conversa, MensagemX, MensagemY

class TestSendMessageFlow(unittest.TestCase):
    def setUp(self):
        app.config['TESTING'] = True
        app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///:memory:'
        self.client = app.test_client()
        with app.app_context():
            db.create_all()
            # Inicializa uma conversa para que a sessão seja configurada
            self.client.get('/')

    def tearDown(self):
        with app.app_context():
            db.session.remove()
            db.drop_all()

    def test_send_message_flow(self):
        # Envia uma mensagem e verifica se as respostas são registradas
        message_payload = {'message': 'Qual é o procedimento para uma licitação?'}
        response = self.client.post('/send_message',
                                    data=json.dumps(message_payload),
                                    content_type='application/json')
        self.assertEqual(response.status_code, 200)
        data = json.loads(response.data)
        self.assertIn('resposta_a', data)
        self.assertIn('resposta_b', data)

        # Verifica se as mensagens foram armazenadas no banco (tabelas MensagemX e MensagemY)
        with app.app_context():
            msgs_x = MensagemX.query.all()
            msgs_y = MensagemY.query.all()
            # Cada mensagem do usuário deve ser registrada em ambas as tabelas
            self.assertGreaterEqual(len(msgs_x), 2)
            self.assertGreaterEqual(len(msgs_y), 2)

if __name__ == '__main__':
    unittest.main()
