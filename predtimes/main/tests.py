from django.test import TestCase, Client
from django.urls import reverse
from .models import Project, TrainingSession, Prediction, ExternalDatabase, ProjectColumn, Model
import json
import configparser
import os
from datetime import datetime, timedelta
from sqlalchemy import create_engine, text
from unittest.mock import patch

class DataViewsTestCase(TestCase):
    def setUp(self):
        self.project = Project.objects.create(name='Test Project', creator='Test User')
        self.client = Client()
        
        # Login the client
        config = configparser.ConfigParser()
        config_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), '..', 'config.ini')
        config.read(config_path)
        user = config.get('auth', 'user', fallback=None)
        password = config.get('auth', 'pass', fallback=None)
        self.client.post(reverse('login'), {'username': user, 'password': password})

        # Create a dummy external database
        self.db_path = 'test_external_db.sqlite'
        self.engine = create_engine(f'sqlite:///{self.db_path}')
        self.create_dummy_data()

        ExternalDatabase.objects.create(
            project=self.project,
            db_type='sqlite',
            host=self.db_path,
            port=0,
            dbname='',
            user='',
            password=''
        )
        ProjectColumn.objects.create(project=self.project, table_name='sales', column_name='sale_date', column_type='datetime')
        ProjectColumn.objects.create(project=self.project, table_name='sales', column_name='amount', column_type='numeric')
        self.model = Model.objects.create(project=self.project, architecture='test_architecture', model_path='', scaler_path='')


    def create_dummy_data(self):
        from sqlalchemy import Table, Column, MetaData, DateTime, Integer
        
        meta = MetaData()
        sales_table = Table('sales', meta,
                            Column('sale_date', DateTime),
                            Column('amount', Integer))
        
        meta.create_all(self.engine)

        dates = [datetime.now() - timedelta(days=i) for i in range(10)]
        data = [{'sale_date': d, 'amount': 100 + i * 10} for i, d in enumerate(dates)]
        
        with self.engine.begin() as connection:
            connection.execute(sales_table.insert(), data)

    def tearDown(self):
        if os.path.exists(self.db_path):
            os.remove(self.db_path)

    def test_get_predictions_data_view(self):
        # Create some dummy predictions
        # Create a training session first
        training_session = TrainingSession.objects.create(
            project=self.project,
            model_architecture='test_architecture'
        )
        # Associate model with the training session
        self.model.training_session = training_session
        self.model.save()
        
        # Create a dummy prediction file
        predictions_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), '..', 'trained')
        os.makedirs(predictions_dir, exist_ok=True)
        file_path = os.path.join(predictions_dir, f"project_{self.project.id}_latest_predictions.json")
        
        dummy_preds = {
            "default": {
                "labels": [(datetime.now() + timedelta(days=i)).isoformat() for i in range(5)],
                "predicted_data": {
                    "amount": [{"x": (datetime.now() + timedelta(days=i)).isoformat(), "y": 200 + i*10} for i in range(5)]
                }
            }
        }
        with open(file_path, 'w') as f:
            json.dump(dummy_preds, f)

        url = reverse('get_predictions_data', args=[self.project.id])
        response = self.client.get(url)
        self.assertEqual(response.status_code, 200)
        response_data = json.loads(response.content)

        self.assertIn('labels', response_data)
        self.assertIn('actual_data', response_data)
        self.assertIn('predicted_data', response_data)
        self.assertEqual(len(response_data['predicted_data']['default']['labels']), 5)
        self.assertEqual(len(response_data['actual_data']['default']['amount']), 10)
        
        # Cleanup the dummy file
        os.remove(file_path)

class TrainingSessionTestCase(TestCase):
    def setUp(self):
        self.project = Project.objects.create(name='Test Project', creator='Test User')
        self.client = Client()
        
        # Login the client
        config = configparser.ConfigParser()
        config_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), '..', 'config.ini')
        config.read(config_path)
        user = config.get('auth', 'user', fallback=None)
        password = config.get('auth', 'pass', fallback=None)
        self.client.post(reverse('login'), {'username': user, 'password': password})


    def test_create_training_session(self):
        training_session = TrainingSession.objects.create(
            project=self.project,
            model_architecture='test_architecture',
        )
        self.assertEqual(training_session.project, self.project)
        self.assertEqual(training_session.model_architecture, 'test_architecture')
        self.assertEqual(training_session.status, 'PENDING')

    def test_save_celery_task_id(self):
        training_session = TrainingSession.objects.create(
            project=self.project,
            model_architecture='test_architecture',
        )
        training_session.celery_task_id = 'test_task_id'
        training_session.save()
        self.assertEqual(training_session.celery_task_id, 'test_task_id')

    @patch('main.views.train_model_task')
    def test_start_training_view(self, mock_train_model_task):
        mock_task = patch('main.views.train_model_task').start()
        mock_task.delay.return_value = type('obj', (object,), {'id': 'test_task_id'})
        
        url = reverse('start_training', args=[self.project.id])
        data = {'architecture': 'test_architecture'}
        response = self.client.post(url, data=json.dumps(data), content_type='application/json')
        
        self.assertEqual(response.status_code, 200)
        
        training_session = TrainingSession.objects.get(project=self.project)
        mock_task.delay.assert_called_once_with(training_session.id)
        
        patch.stopall()

    def test_get_training_status_view(self):
        training_session = TrainingSession.objects.create(
            project=self.project,
            model_architecture='test_architecture',
            celery_task_id='test_task_id'
        )
        url = reverse('get_training_status', args=[training_session.id])
        response = self.client.get(url)
        self.assertEqual(response.status_code, 200)
        response_data = json.loads(response.content)
        self.assertIn('celery_task_id', response_data)
        self.assertEqual(response_data['celery_task_id'], 'test_task_id')