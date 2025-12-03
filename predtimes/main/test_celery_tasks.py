"""
This file contains tests for the Celery tasks in the 'main' application.
"""
from django.test import TestCase, Client
from django.urls import reverse
from .models import Project, TrainingSession, Model, ExternalDatabase, SelectedTable, ProjectColumn
import os
from sqlalchemy import create_engine, text
from datetime import datetime, timedelta
from .tasks import train_model_task, generate_predictions_task
import configparser
import json

class TrainModelTaskTestCase(TestCase):
    """
    Test case for the train_model_task Celery task.
    """
    def setUp(self):
        # Create a project
        self.project = Project.objects.create(name='Test Project for Training', creator='Test User')

        # Create a dummy external database
        self.db_path = 'test_external_db_for_training.sqlite'
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
        SelectedTable.objects.create(project=self.project, table_name='sales')
        
        self.client = Client()
        config = configparser.ConfigParser()
        config_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), '..', 'config.ini')
        config.read(config_path)
        user = config.get('auth', 'user', fallback=None)
        password = config.get('auth', 'pass', fallback=None)
        self.client.post(reverse('login'), {'username': user, 'password': password})


    def create_dummy_data(self):
        from sqlalchemy import Table, Column, MetaData, DateTime, Integer
        
        meta = MetaData()
        sales_table = Table('sales', meta,
                            Column('sale_date', DateTime),
                            Column('amount', Integer),
                            Column('category', Integer))
        
        meta.create_all(self.engine)

        dates = [datetime.now() - timedelta(days=i) for i in range(20)]
        data = [{'sale_date': d, 'amount': 100 + i * 10, 'category': i % 3} for i, d in enumerate(dates)]
        
        with self.engine.begin() as connection:
            connection.execute(sales_table.insert(), data)

    def tearDown(self):
        if hasattr(self, 'training_session'):
            trained_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), '..', 'trained')
            if os.path.exists(trained_dir):
                for f in os.listdir(trained_dir):
                    if f.startswith(f"session_{self.training_session.id}"):
                        os.remove(os.path.join(trained_dir, f))
        if os.path.exists(self.db_path):
            os.remove(self.db_path)


    def test_train_model_task_success(self):
        # Setup columns
        ProjectColumn.objects.create(project=self.project, table_name='sales', column_name='sale_date', column_type='datetime')
        ProjectColumn.objects.create(project=self.project, table_name='sales', column_name='amount', column_type='numeric')
        
        # Create a training session
        self.training_session = TrainingSession.objects.create(
            project=self.project,
            model_architecture='rnn'
        )

        # Run the task synchronously
        result = train_model_task.run(self.training_session.id)

        # Refresh session from DB
        self.training_session.refresh_from_db()

        self.assertEqual(self.training_session.status, 'SUCCESS')
        self.assertEqual(result['status'], 'SUCCESS')

        # Check for model in DB
        self.assertTrue(Model.objects.filter(training_session=self.training_session).exists())
        model_record = Model.objects.get(training_session=self.training_session)

        # Check if files were created
        self.assertTrue(os.path.exists(model_record.model_path))
        self.assertTrue(os.path.exists(model_record.scaler_path))

    def test_train_model_task_failure_no_columns(self):
        # Create a training session without setting up columns
        self.training_session = TrainingSession.objects.create(
            project=self.project,
            model_architecture='rnn'
        )

        # Run the task synchronously and expect a ValueError
        with self.assertRaises(ValueError):
            train_model_task.run(self.training_session.id)

        # Refresh session from DB
        self.training_session.refresh_from_db()
        
        self.assertEqual(self.training_session.status, 'FAILURE')


class GeneratePredictionsTaskTestCase(TestCase):
    """
    Test case for the generate_predictions_task Celery task.
    """
    def setUp(self):
        # Create a project
        self.project = Project.objects.create(name='Test Project for Prediction', creator='Test User')

        # Create a dummy external database
        self.db_path = 'test_external_db_for_prediction.sqlite'
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
        SelectedTable.objects.create(project=self.project, table_name='sales')
        ProjectColumn.objects.create(project=self.project, table_name='sales', column_name='sale_date', column_type='datetime')
        ProjectColumn.objects.create(project=self.project, table_name='sales', column_name='amount', column_type='numeric')
        
        # Create and run a training session to have a model
        self.training_session = TrainingSession.objects.create(
            project=self.project,
            model_architecture='rnn'
        )
        train_model_task.run(self.training_session.id)
        
        self.client = Client()
        config = configparser.ConfigParser()
        config_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), '..', 'config.ini')
        config.read(config_path)
        user = config.get('auth', 'user', fallback=None)
        password = config.get('auth', 'pass', fallback=None)
        self.client.post(reverse('login'), {'username': user, 'password': password})

    def create_dummy_data(self):
        from sqlalchemy import Table, Column, MetaData, DateTime, Integer
        
        meta = MetaData()
        sales_table = Table('sales', meta,
                            Column('sale_date', DateTime),
                            Column('amount', Integer),
                            Column('category', Integer))
        
        meta.create_all(self.engine)

        dates = [datetime.now() - timedelta(days=i) for i in range(20)]
        data = [{'sale_date': d, 'amount': 100 + i * 10, 'category': i % 3} for i, d in enumerate(dates)]
        
        with self.engine.begin() as connection:
            connection.execute(sales_table.insert(), data)

    def tearDown(self):
        if os.path.exists(self.db_path):
            os.remove(self.db_path)
        
        trained_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), '..', 'trained')
        prediction_file = os.path.join(trained_dir, f"project_{self.project.id}_latest_predictions.json")
        if os.path.exists(prediction_file):
            os.remove(prediction_file)
            
        if hasattr(self, 'training_session'):
            if os.path.exists(trained_dir):
                for f in os.listdir(trained_dir):
                    if f.startswith(f"session_{self.training_session.id}"):
                        os.remove(os.path.join(trained_dir, f))

    def test_generate_predictions_task_success(self):
        # Run the prediction task
        result = generate_predictions_task.run(self.project.id)

        # Check the result message
        self.assertIn("Generated 10 predictions", result)

        # Check if prediction file was created
        trained_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), '..', 'trained')
        prediction_file = os.path.join(trained_dir, f"project_{self.project.id}_latest_predictions.json")
        self.assertTrue(os.path.exists(prediction_file))

        # Check the content of the prediction file
        with open(prediction_file, 'r') as f:
            predictions_data = json.load(f)
        
        self.assertIn('default', predictions_data)
        self.assertIn('labels', predictions_data['default'])
        self.assertIn('predicted_data', predictions_data['default'])
        self.assertEqual(len(predictions_data['default']['labels']), 10)

    def test_generate_predictions_task_missing_model_file(self):
        # Get the path to the model file
        model_record = Model.objects.get(training_session=self.training_session)
        model_path = model_record.model_path

        # Delete the model file
        os.remove(model_path)

        # Run the prediction task and expect a FileNotFoundError
        with self.assertRaises(FileNotFoundError):
            generate_predictions_task.run(self.project.id)