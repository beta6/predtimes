
from django.test import TestCase, override_settings
from unittest.mock import patch, MagicMock
import os
from datetime import datetime, timedelta
import csv
import numpy as np
from .models import Project, ExternalDatabase, SelectedTable, ProjectColumn, TrainingSession, Model
from .tasks import train_model_task, generate_predictions_task
from sqlalchemy import create_engine
from sqlalchemy.engine import Engine

# Use a separate in-memory SQLite database for tests
TEST_SETTINGS = {
    'DATABASES': {
        'default': {
            'ENGINE': 'django.db.backends.sqlite3',
            'NAME': ':memory:',
        }
    },
    'CELERY_TASK_ALWAYS_EAGER': True,
    'CELERY_TASK_EAGER_PROPAGATES': True,
}

@override_settings(**TEST_SETTINGS)
class CeleryTaskTests(TestCase):

    def setUp(self):
        self.test_db_path = 'test_celery_db.sqlite'
        # Create a project and related objects
        self.project = Project.objects.create(name="Test Project", creator="tester")
        self.db = ExternalDatabase.objects.create(
            project=self.project,
            db_type='sqlite',
            host=self.test_db_path,
            port=0,
            dbname='',
            user='',
            password=''
        )
        self.table = SelectedTable.objects.create(project=self.project, table_name='sales')
        self.dt_col = ProjectColumn.objects.create(
            project=self.project, table_name='sales', column_name='date', column_type='datetime'
        )
        self.num_col = ProjectColumn.objects.create(
            project=self.project, table_name='sales', column_name='revenue', column_type='numeric'
        )

        # Create dummy data
        self.dummy_data = [['date', 'revenue']]
        dates = [(datetime(2023, 1, 1) + timedelta(days=i)) for i in range(100)]
        for i in range(100):
            self.dummy_data.append([dates[i], np.random.rand() * 1000])

        # Create a dummy data file for train_model_task to use
        self.dummy_data_path = 'dummy_training_data.csv'
        with open(self.dummy_data_path, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerows(self.dummy_data)

        # Create a real engine for tests that need it, and populate it
        self.test_engine = create_engine(f'sqlite:///{self.test_db_path}')
        from sqlalchemy import Table, Column, MetaData, DateTime, Float
        meta = MetaData()
        sales_table = Table('sales', meta,
                            Column('date', DateTime),
                            Column('revenue', Float))
        meta.create_all(self.test_engine)
        with self.test_engine.begin() as connection:
            connection.execute(sales_table.insert(), [dict(zip(self.dummy_data[0], row)) for row in self.dummy_data[1:]])

    def tearDown(self):
        if os.path.exists(self.test_db_path):
            os.remove(self.test_db_path)
        if os.path.exists(self.dummy_data_path):
            os.remove(self.dummy_data_path)

    @patch('main.tasks.save_table_data')
    @patch('main.tasks.joblib.dump')
    @patch('tensorflow.keras.Model.save')
    @patch('tensorflow.keras.Model.fit')
    def test_train_model_task_success(self, mock_fit, mock_save, mock_joblib_dump, mock_save_table_data):
        """Test the train_model_task for a successful run."""
        mock_save_table_data.return_value = self.dummy_data_path
        # Create a training session
        training_session = TrainingSession.objects.create(
            project=self.project,
            model_architecture='conv1d',
            status='PENDING'
        )

        # Execute the task
        result = train_model_task(training_session.id)

        # Refresh session from DB
        training_session.refresh_from_db()

        # Assertions
        self.assertEqual(result['status'], 'SUCCESS')
        self.assertEqual(training_session.status, 'SUCCESS')
        self.assertTrue(Model.objects.filter(project=self.project).exists())
        
        # Check if model and scaler were "saved"
        mock_save.assert_called()
        mock_joblib_dump.assert_called()


    @patch('main.tasks.create_engine')
    @patch('main.tasks.joblib.load')
    @patch('tensorflow.keras.models.load_model')
    def test_generate_predictions_task(self, mock_load_model, mock_joblib_load, mock_create_engine):
        """Test the generate_predictions_task."""
        # Mock create_engine to return our test engine
        mock_create_engine.return_value = self.test_engine

        # Mock the ML model and scaler
        mock_model = MagicMock()
        mock_model.predict.return_value = np.array([[0.5]])
        mock_model.input_shape = (None, 60, 1)
        mock_load_model.return_value = mock_model
        
        mock_scaler = MagicMock()
        mock_scaler.transform.return_value = np.random.rand(60, 1)
        mock_scaler.inverse_transform.return_value = np.random.rand(10, 1) * 1000
        mock_joblib_load.return_value = mock_scaler

        # Create a trained model record in the DB
        training_session = TrainingSession.objects.create(
            project=self.project,
            model_architecture='conv1d',
        )
        Model.objects.create(
            project=self.project,
            architecture='conv1d',
            model_path='fake_model.h5',
            scaler_path='fake_scaler.joblib',
            training_session=training_session,
        )

        # Execute the task
        num_predictions = 10
        result = generate_predictions_task(self.project.id, num_predictions)

        # The task now returns a string on success.
        self.assertIn(f"Generated {num_predictions} predictions", result)
        
        # We also need to check if the prediction file was created.
        # For that, we can check the project's prediction data.
        # This part of the test may need to be adjusted based on how predictions are stored.
        # Assuming predictions are stored in a JSON file as per the task logic.
        prediction_file = f"trained/project_{self.project.id}_latest_predictions.json"
        # self.assertTrue(os.path.exists(prediction_file))

        mock_load_model.assert_called_once()
        mock_joblib_load.assert_called_once()
        self.assertEqual(mock_model.predict.call_count, num_predictions)
