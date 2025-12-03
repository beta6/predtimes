"""
This file contains the unit and integration tests for the 'main' application.
"""
from django.test import TestCase, Client
from django.urls import reverse
from .models import Project, TrainingSession, ExternalDatabase, ProjectColumn, Model
import json
import configparser
import os
from datetime import datetime, timedelta
from sqlalchemy import create_engine, text
from unittest.mock import patch

class DataViewsTestCase(TestCase):
    """
    Test case for views related to data handling, such as fetching predictions
    and project details.
    """
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

    def test_get_predictions_data_view_no_file(self):
        # Ensure no prediction file exists
        predictions_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), '..', 'trained')
        file_path = os.path.join(predictions_dir, f"project_{self.project.id}_latest_predictions.json")
        if os.path.exists(file_path):
            os.remove(file_path)

        url = reverse('get_predictions_data', args=[self.project.id])
        response = self.client.get(url)
        self.assertEqual(response.status_code, 200)
        response_data = json.loads(response.content)

        self.assertIn('predicted_data', response_data)
        self.assertEqual(response_data['predicted_data'], {})

    def test_home_view(self):
        # Create some projects to test ordering
        project1 = Project.objects.create(name='Project 1', creator='Test User')
        project2 = Project.objects.create(name='Project 2', creator='Test User')

        url = reverse('home')
        response = self.client.get(url)
        self.assertEqual(response.status_code, 200)
        self.assertTemplateUsed(response, 'home.html')

        # Check that the projects are in the context and ordered correctly
        self.assertIn('projects', response.context)
        projects_in_context = list(response.context['projects'])
        self.assertEqual(len(projects_in_context), 3) # Including the one from setUp
        self.assertEqual(projects_in_context[0], project2)
        self.assertEqual(projects_in_context[1], project1)
        self.assertEqual(projects_in_context[2], self.project)

    def test_project_detail_view(self):
        url = reverse('project_detail', args=[self.project.id])
        response = self.client.get(url)
        self.assertEqual(response.status_code, 200)
        self.assertTemplateUsed(response, 'project_detail.html')
        self.assertEqual(response.context['project'], self.project)
        self.assertIn('get_data_task_id', response.context)
        self.assertIn('selected_datetime_columns', response.context)
        self.assertIn('selected_numeric_columns', response.context)
        self.assertIn('selected_multigroup_columns', response.context)
        self.assertIn('serialized_selected_tables', response.context)

    @patch('main.views.get_db_engine_from_details')
    def test_test_db_connection_view(self, mock_get_db_engine):
        # Mock a successful connection
        mock_engine = mock_get_db_engine.return_value
        mock_connection = mock_engine.connect.return_value
        
        url = reverse('test_db_connection')
        form_data = {
            'db_type': 'sqlite', 'host': 'test.db', 'port': 0,
            'dbname': 'test', 'user': 'user', 'password': 'password'
        }
        response = self.client.post(url, form_data)
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json()['status'], 'ok')

        # Mock a failed connection
        from sqlalchemy.exc import OperationalError
        mock_get_db_engine.side_effect = OperationalError("Simulated error", {}, None)
        response = self.client.post(url, form_data)
        self.assertEqual(response.status_code, 400)
        self.assertEqual(response.json()['status'], 'error')

    def test_download_predictions_csv_view(self):
        # Create a dummy prediction file
        predictions_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), '..', 'trained')
        os.makedirs(predictions_dir, exist_ok=True)
        file_path = os.path.join(predictions_dir, f"project_{self.project.id}_latest_predictions.json")
        
        dummy_preds = {
            "default": {
                "labels": ["2025-12-02T12:00:00"],
                "predicted_data": {
                    "amount": [{"x": "2025-12-02T12:00:00", "y": 200}]
                }
            }
        }
        with open(file_path, 'w') as f:
            json.dump(dummy_preds, f)

        url = reverse('download_predictions_csv', args=[self.project.id])
        response = self.client.get(url)
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response['Content-Type'], 'text/csv')
        
        content = response.content.decode('utf-8')
        self.assertIn('group,timestamp,predicted_amount', content)
        self.assertIn('default,2025-12-02T12:00:00,200', content)

        # Cleanup the dummy file
        os.remove(file_path)

        # Test case where prediction file does not exist
        response = self.client.get(url)
        self.assertEqual(response.status_code, 404)

    @patch('main.views.SelectedTable.objects.create')
    def test_project_wizard_step3_rollback(self, mock_create_selected_table):
        # Configure the mock to raise an exception
        mock_create_selected_table.side_effect = Exception("Simulated error on purpose")

        # Set up session data for the wizard
        session = self.client.session
        session['wizard_db_data'] = {
            'db_type': 'sqlite', 'host': 'test.db', 'port': 0,
            'dbname': 'test', 'user': 'user', 'password': 'password'
        }
        session['wizard_selected_tables'] = ['table1', 'table2']
        session.save()

        # Data for the project form
        project_data = {'name': 'Rollback Test Project', 'creator': 'Test User'}

        # Count objects before the request
        projects_before = Project.objects.count()
        dbs_before = ExternalDatabase.objects.count()

        # Call the view
        url = reverse('wizard_step3')
        self.client.post(url, project_data)

        # Assert that no new objects were created
        self.assertEqual(Project.objects.count(), projects_before)
        self.assertEqual(ExternalDatabase.objects.count(), dbs_before)

    def test_save_column_selection_rollback(self):
        # Create an initial column that should be preserved after the rollback
        initial_column = ProjectColumn.objects.create(
            project=self.project,
            table_name='sales',
            column_name='old_amount',
            column_type='numeric'
        )

        with patch('main.views.ProjectColumn.objects.create') as mock_create_project_column:
            # Configure the mock to raise an exception
            mock_create_project_column.side_effect = Exception("Simulated error on purpose")

            # Data for the save_column_selection view
            post_data = {
                'columns': ['sales.new_amount'],
                'type': 'numeric'
            }

            # Count objects before the request
            columns_before = ProjectColumn.objects.filter(project=self.project, column_type='numeric').count()

            # Call the view
            url = reverse('save_column_selection', args=[self.project.id])
            response = self.client.post(url, data=json.dumps(post_data), content_type='application/json')

            # Assert that the view returns a 500 status code
            self.assertEqual(response.status_code, 500)

            # Assert that the number of columns is the same as before
            columns_after = ProjectColumn.objects.filter(project=self.project, column_type='numeric').count()
            self.assertEqual(columns_after, columns_before)

            # Assert that the initial column still exists
            self.assertTrue(ProjectColumn.objects.filter(id=initial_column.id).exists())

class TrainingSessionTestCase(TestCase):
    """
    Test case for the TrainingSession model and related views.
    """
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

class ProjectDeletionTestCase(TestCase):
    """
    Test case for project deletion and the cascading deletion of related objects.
    """
    def setUp(self):
        self.project = Project.objects.create(name='Test Project for Deletion', creator='Test User')
        self.db = ExternalDatabase.objects.create(project=self.project, db_type='sqlite', dbname='test.db', port=0)
        self.model = Model.objects.create(project=self.project, architecture='test_arch', model_path='', scaler_path='')
        self.training_session = TrainingSession.objects.create(project=self.project, model_architecture='test_arch')
        
        # Create a user and log in
        self.client = Client()
        config = configparser.ConfigParser()
        config_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), '..', 'config.ini')
        config.read(config_path)
        user = config.get('auth', 'user', fallback=None)
        password = config.get('auth', 'pass', fallback=None)
        self.client.post(reverse('login'), {'username': user, 'password': password})

    def test_delete_project_cascades(self):
        # Verify that all objects exist before deletion
        self.assertTrue(Project.objects.filter(id=self.project.id).exists())
        self.assertTrue(ExternalDatabase.objects.filter(project=self.project).exists())
        self.assertTrue(Model.objects.filter(project=self.project).exists())
        self.assertTrue(TrainingSession.objects.filter(project=self.project).exists())

        # Call the delete_project view
        url = reverse('delete_project', args=[self.project.id])
        response = self.client.post(url)

        # Check that the user is redirected to the home page
        self.assertRedirects(response, reverse('home'))

        # Verify that the project and all associated objects are deleted
        self.assertFalse(Project.objects.filter(id=self.project.id).exists())
        self.assertFalse(ExternalDatabase.objects.filter(project=self.project).exists())
        self.assertFalse(Model.objects.filter(project=self.project).exists())
        self.assertFalse(TrainingSession.objects.filter(project=self.project).exists())

class AuthenticationTestCase(TestCase):
    """
    Test case for user authentication (login and logout).
    """
    def setUp(self):
        self.client = Client()
        self.login_url = reverse('login')
        self.home_url = reverse('home')
        
        config = configparser.ConfigParser()
        config_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), '..', 'config.ini')
        config.read(config_path)
        self.user = config.get('auth', 'user', fallback='admin')
        self.password = config.get('auth', 'pass', fallback='password')

    def test_login_view(self):
        # Test GET request
        response = self.client.get(self.login_url)
        self.assertEqual(response.status_code, 200)
        self.assertTemplateUsed(response, 'login.html')

        # Test POST with incorrect credentials
        response = self.client.post(self.login_url, {'username': 'wronguser', 'password': 'wrongpassword'})
        self.assertEqual(response.status_code, 200)
        self.assertTemplateUsed(response, 'login.html')
        self.assertContains(response, 'Invalid credentials')

        # Test POST with correct credentials
        response = self.client.post(self.login_url, {'username': self.user, 'password': self.password})
        self.assertRedirects(response, self.home_url)
        self.assertTrue(self.client.session['is_logged_in'])

    def test_logout_view(self):
        # Log in first
        self.client.post(self.login_url, {'username': self.user, 'password': self.password})
        self.assertTrue(self.client.session['is_logged_in'])

        # Test logout
        logout_url = reverse('logout')
        response = self.client.get(logout_url)
        self.assertRedirects(response, self.login_url)
        self.assertNotIn('is_logged_in', self.client.session)

class ProjectWizardTestCase(TestCase):
    """
    Test case for the project creation wizard.
    """
    def setUp(self):
        self.client = Client()
        
        # Login the client
        config = configparser.ConfigParser()
        config_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), '..', 'config.ini')
        config.read(config_path)
        user = config.get('auth', 'user', fallback='admin')
        password = config.get('auth', 'pass', fallback='password')
        self.client.post(reverse('login'), {'username': user, 'password': password})

        self.step1_url = reverse('wizard_step1')
        self.step2_url = reverse('wizard_step2')

    def test_project_wizard_step1(self):
        # Test GET request
        response = self.client.get(self.step1_url)
        self.assertEqual(response.status_code, 200)
        self.assertTemplateUsed(response, 'project_wizard_step1.html')

        # Test POST request with valid data
        form_data = {
            'db_type': 'sqlite',
            'host': 'test.db',
            'port': 0,
            'dbname': 'test',
            'user': 'user',
            'password': 'password'
        }
        response = self.client.post(self.step1_url, form_data)
        self.assertRedirects(response, self.step2_url)
        self.assertEqual(self.client.session['wizard_db_data'], form_data)

    @patch('sqlalchemy.inspect')
    def test_project_wizard_step2(self, mock_inspect):
        # Mock the database inspection
        mock_inspector = mock_inspect.return_value
        mock_inspector.get_table_names.return_value = ['table1', 'table2']

        # Set up session data from step 1
        session = self.client.session
        session['wizard_db_data'] = {
            'db_type': 'sqlite', 'host': 'test.db', 'port': 0,
            'dbname': 'test', 'user': 'user', 'password': 'password'
        }
        session.save()

        # Test GET request
        response = self.client.get(self.step2_url)
        self.assertEqual(response.status_code, 200)
        self.assertTemplateUsed(response, 'project_wizard_step2.html')
        self.assertEqual(response.context['available_tables'], ['table1', 'table2'])

        # Test POST request with valid data
        response = self.client.post(self.step2_url, {'selected_tables': ['table1']})
        self.assertRedirects(response, reverse('wizard_step3'))
        self.assertEqual(self.client.session['wizard_selected_tables'], ['table1'])