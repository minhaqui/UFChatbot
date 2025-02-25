import unittest
import json
from app.app import app, db, MensagemX, MensagemY, Conversa

class TestHistorico(unittest.TestCase):

    def setUp(self):
        app.config['TESTING'] = True
        # Utiliza um banco de dados em memória para testes isolados
        app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///:memory:'
        self.client = app.test_client()
        with app.app_context():
            db.create_all()

    def tearDown(self):
        with app.app_context():
            db.session.remove()
            db.drop_all()

    def test_context_saved(self):
        # Inicializa uma nova conversa
        response = self.client.get('/')
        self.assertEqual(response.status_code, 200)
        # Captura o ID da conversa a partir da sessão
        with self.client.session_transaction() as sess:
            conversation_id = sess.get('conversa_id')
        # Envia uma mensagem e espera que ela seja armazenada
        message_payload = {'message': 'Mensagem de teste'}
        response = self.client.post(
            '/send_message',
            data=json.dumps(message_payload),
            content_type='application/json'
        )
        self.assertEqual(response.status_code, 200)
        # Verifica no banco de dados se as mensagens foram salvas em MensagemX (Chat A)
        with app.app_context():
            msgs = MensagemX.query.filter_by(conversa_id=conversation_id).all()
            # Espera que pelo menos duas mensagens sejam salvas: uma do usuário e outra do modelo
            self.assertGreaterEqual(len(msgs), 2)

    def test_context_reset_on_reset(self):
        # Inicializa uma nova conversa
        response = self.client.get('/')
        self.assertEqual(response.status_code, 200)
        with self.client.session_transaction() as sess:
            conversation_id_before = sess.get('conversa_id')
        # Envia uma mensagem para criar histórico
        message_payload = {'message': 'Mensagem para reset'}
        self.client.post(
            '/send_message',
            data=json.dumps(message_payload),
            content_type='application/json'
        )
        # Confirma que há mensagens associadas ao conversation_id atual
        with app.app_context():
            msgs_before = MensagemX.query.filter_by(conversa_id=conversation_id_before).all()
            self.assertGreater(len(msgs_before), 0)
        # Chama a rota de reset
        response = self.client.post('/reset')
        self.assertEqual(response.status_code, 200)
        # Verifica que o conversation_id foi alterado após o reset
        with self.client.session_transaction() as sess:
            conversation_id_after = sess.get('conversa_id')
        self.assertNotEqual(conversation_id_before, conversation_id_after)
        # Confirma que as mensagens associadas ao antigo conversation_id foram removidas
        with app.app_context():
            msgs_after = MensagemX.query.filter_by(conversa_id=conversation_id_before).all()
            self.assertEqual(len(msgs_after), 0)

    def test_context_reset_on_evaluation(self):
        # Inicializa uma nova conversa
        response = self.client.get('/')
        self.assertEqual(response.status_code, 200)
        with self.client.session_transaction() as sess:
            conversation_id_before = sess.get('conversa_id')
        # Envia uma mensagem para criar histórico
        message_payload = {'message': 'Mensagem para avaliação'}
        self.client.post(
            '/send_message',
            data=json.dumps(message_payload),
            content_type='application/json'
        )
        # Registra uma avaliação (o que deve resetar o contexto)
        eval_payload = {
            'winner': 'Chat A',
            'nome': 'Teste',
            'email': 'teste@example.com',
            'proficiencia': 'Intermediario'
        }
        response = self.client.post(
            '/evaluate',
            data=json.dumps(eval_payload),
            content_type='application/json'
        )
        self.assertEqual(response.status_code, 200)
        # Verifica que o conversation_id mudou após a avaliação
        with self.client.session_transaction() as sess:
            conversation_id_after = sess.get('conversa_id')
        self.assertNotEqual(conversation_id_before, conversation_id_after)

if __name__ == '__main__':
    unittest.main()
