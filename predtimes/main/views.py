from django.shortcuts import render, redirect, get_object_or_404
from django.urls import reverse
import configparser
import os
from django.http import JsonResponse
from django.views.decorators.http import require_POST
from .forms import ExternalDatabaseForm
from functools import wraps
from django.http import HttpResponse
import pandas as pd
from .db_utils import get_db_engine_from_details
from .hints import HINTS

def login_required(view_func):
    """Custom decorator to check for session-based login."""
    @wraps(view_func)
    def _wrapped_view(request, *args, **kwargs):
        if not request.session.get('is_logged_in', False):
            return redirect(f"{reverse('login')}?next={request.path}")
        return view_func(request, *args, **kwargs)
    return _wrapped_view

@require_POST
@login_required
def test_db_connection(request):
    """
    Tests a connection to an external database using data from a POST request.
    This requires SQLAlchemy and appropriate DB drivers to be installed.
    e.g., pip install SQLAlchemy psycopg2-binary mysqlclient
    """
    form = ExternalDatabaseForm(request.POST)
    if not form.is_valid():
        return JsonResponse({'status': 'error', 'message': 'Invalid form data.'}, status=400)

    data = form.cleaned_data
    
    try:
        from sqlalchemy.exc import OperationalError
        engine = get_db_engine_from_details(
            db_type=data['db_type'],
            user=data['user'],
            password=data['password'],
            host=data['host'],
            port=data['port'],
            dbname=data['dbname']
        )
        with engine.connect() as connection:
            return JsonResponse({'status': 'ok', 'message': 'Connection successful!'})
    except (ImportError, ValueError) as e:
        return JsonResponse({'status': 'error', 'message': f'Unsupported database type or missing driver: {e}'}, status=500)
    except OperationalError as e:
        return JsonResponse({'status': 'error', 'message': f'Connection failed: {e}'}, status=400)
    except Exception as e:
        return JsonResponse({'status': 'error', 'message': f'An unexpected error occurred: {e}'}, status=500)


def login_view(request):
    if request.session.get('is_logged_in', False):
        return redirect(reverse('home'))

    error = None
    if request.method == 'POST':
        config = configparser.ConfigParser()
        config_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), '..', 'config.ini')
        config.read(config_path)
        
        user = config.get('auth', 'user', fallback=None)
        password = config.get('auth', 'pass', fallback=None)

        if request.POST.get('username') == user and request.POST.get('password') == password:
            request.session['is_logged_in'] = True
            request.session['username'] = user
            return redirect(reverse('home'))
        else:
            error = "Invalid credentials. Please try again."

    return render(request, 'login.html', {'error': error})

def logout_view(request):
    request.session.flush()
    return redirect(reverse('login'))

@login_required
def home_view(request):
    username = request.session.get('username', 'User')
    projects = Project.objects.all().order_by('-created_at')
    context = {
        'user': username,
        'projects': projects,
        'hint': HINTS.get('home')
    }
    return render(request, 'home.html', context)

@login_required
def project_wizard_step1(request):
    if request.method == 'POST':
        form = ExternalDatabaseForm(request.POST)
        if form.is_valid():
            # Store form data in session and move to the next step
            request.session['wizard_db_data'] = form.cleaned_data
            return redirect(reverse('wizard_step2'))
    else:
        form = ExternalDatabaseForm()
    
    context = {
        'form': form,
        'hint': HINTS.get('project_wizard_step1')
    }
    return render(request, 'project_wizard_step1.html', context)

from django.contrib import messages

@login_required
def project_wizard_step2(request):
    db_data = request.session.get('wizard_db_data')
    if not db_data:
        messages.error(request, "Database connection data not found. Please start from step 1.")
        return redirect(reverse('wizard_step1'))

    try:
        from sqlalchemy import inspect
        
        engine = get_db_engine_from_details(
            db_type=db_data['db_type'],
            user=db_data['user'],
            password=db_data['password'],
            host=db_data['host'],
            port=db_data['port'],
            dbname=db_data['dbname']
        )
        inspector = inspect(engine)
        available_tables = inspector.get_table_names()

    except Exception as e:
        messages.error(request, f"Could not connect to the database to fetch tables: {e}")
        return redirect(reverse('wizard_step1'))

    if request.method == 'POST':
        selected_tables = request.POST.getlist('selected_tables')
        if not selected_tables:
            messages.error(request, "You must select at least one table.")
        else:
            request.session['wizard_selected_tables'] = selected_tables
            return redirect(reverse('wizard_step3'))

    context = {
        'available_tables': available_tables,
        'hint': HINTS.get('project_wizard_step2')
    }
    return render(request, 'project_wizard_step2.html', context)

from .forms import ExternalDatabaseForm, ProjectForm
from .models import Project, ExternalDatabase, SelectedTable, ProjectColumn, TrainingSession
from django.db import transaction
import json
from .tasks import train_model_task, generate_predictions_task, get_table_sample_data_task
from .somecalls import _get_actual_data_for_chart
from celery.result import AsyncResult

@login_required
def get_prediction_status(request, task_id):
    """
    Returns the current status of a Celery task.
    """
    task_result = AsyncResult(task_id)
    
    if task_result.failed():
        result = {
            'task_id': task_id,
            'status': task_result.status,
            'result': str(task_result.result), # Convert exception to string
        }
    else:
        result = {
            'task_id': task_id,
            'status': task_result.status,
            'result': task_result.result,
        }
    return JsonResponse(result)


@login_required
@require_POST
def save_column_selection(request, project_id):
    try:
        project = Project.objects.get(id=project_id)
        data = json.loads(request.body)
        columns = data.get('columns', [])
        column_type = data.get('type', '')

        if not column_type or column_type not in ['datetime', 'numeric', 'multigroup']:
            return JsonResponse({'status': 'error', 'message': 'Invalid column type specified.'}, status=400)

        with transaction.atomic():
            # Delete old columns of the same type for this project
            ProjectColumn.objects.filter(project=project, column_type=column_type).delete()

            # Create new ones
            for col_str in columns:
                table_name, column_name = col_str.split('.', 1)
                ProjectColumn.objects.create(
                    project=project,
                    table_name=table_name,
                    column_name=column_name,
                    column_type=column_type
                )
        
        return JsonResponse({'status': 'ok', 'message': 'Selection saved successfully!'})
    except Project.DoesNotExist:
        return JsonResponse({'status': 'error', 'message': 'Project not found.'}, status=404)
    except Exception as e:
        return JsonResponse({'status': 'error', 'message': str(e)}, status=500)

@login_required
@require_POST
def start_training(request, project_id):
    """Triggers a Celery task to start model training."""
    project = get_object_or_404(Project, id=project_id)
    
    try:
        data = json.loads(request.body)
        architecture = data.get('architecture')
        if not architecture:
            return JsonResponse({'status': 'error', 'message': 'Model architecture not specified.'}, status=400)

        # Create a TrainingSession record
        training_session = TrainingSession.objects.create(
            project=project,
            model_architecture=architecture,
            status='PENDING',
            details='Training task queued.'
        )

        # Call the training task directly
        task = train_model_task.delay(training_session.id)
        
        # Save the Celery task ID to the session
        training_session.celery_task_id = task.id
        training_session.save()
        
        message = f"Training has been initiated for project '{project.name}' with {architecture} architecture. You can monitor its progress."
        return JsonResponse({'status': 'ok', 'message': message, 'training_session_id': training_session.id, 'celery_task_id': training_session.celery_task_id})
    
    except json.JSONDecodeError:
        return JsonResponse({'status': 'error', 'message': 'Invalid JSON in request body.'}, status=400)
    except Exception as e:
        return JsonResponse({'status': 'error', 'message': f'An unexpected error occurred: {e}'}, status=500)

@login_required
def get_training_status(request, training_session_id):
    """
    Returns the current status of a training session.
    """
    try:
        training_session = TrainingSession.objects.get(id=training_session_id)
        return JsonResponse({
            'status': training_session.status,
            'details': training_session.details,
            'model_architecture': training_session.model_architecture,
            'created_at': training_session.created_at.isoformat(),
            'updated_at': training_session.updated_at.isoformat(),
            'celery_task_id': training_session.celery_task_id,
        })
    except TrainingSession.DoesNotExist:
        return JsonResponse({'status': 'error', 'message': 'Training session not found.'}, status=404)
    except Exception as e:
        return JsonResponse({'status': 'error', 'message': f'An unexpected error occurred: {e}'}, status=500)

@login_required
def get_table_data(request, task_id):
    """
    Returns the result of a get_table_sample_data_task.
    """
    task_result = AsyncResult(task_id)
    if task_result.ready():
        if task_result.successful():
            result = task_result.get()
            return JsonResponse({'status': 'SUCCESS', 'data': result})
        else:
            # Task failed, get the exception
            # The result of a failed task is the exception object
            result = task_result.result
            return JsonResponse({'status': 'FAILURE', 'message': str(result)})
    else:
        return JsonResponse({'status': task_result.status})

@login_required
@require_POST
def trigger_table_data_refresh(request, project_id):
    """
    Triggers a Celery task to refresh table sample data for a project
    and returns the task ID.
    """
    project = get_object_or_404(Project, id=project_id)
    task = get_table_sample_data_task.delay(project.id)
    return JsonResponse({'status': 'ok', 'task_id': task.id})

@login_required
def project_detail(request, project_id):
    project = get_object_or_404(Project, id=project_id)
    
    # Trigger the Celery task to get table sample data
    task = get_table_sample_data_task.delay(project.id)
    
    selected_datetime_columns = [f"{c.table_name}.{c.column_name}" for c in project.columns.filter(column_type='datetime')]
    selected_numeric_columns = [f"{c.table_name}.{c.column_name}" for c in project.columns.filter(column_type='numeric')]
    selected_multigroup_columns = [f"{c.table_name}.{c.column_name}" for c in project.columns.filter(column_type='multigroup')]
    
    serialized_selected_tables = [{'table_name': st.table_name} for st in project.selected_tables.all()]
    
    context = {
        'project': project,
        'get_data_task_id': task.id,
        'selected_datetime_columns': selected_datetime_columns,
        'selected_numeric_columns': selected_numeric_columns,
        'selected_multigroup_columns': selected_multigroup_columns,
        'serialized_selected_tables': serialized_selected_tables,
        'hint': HINTS.get('project_detail')
    }
    return render(request, 'project_detail.html', context)

@login_required
@require_POST
def generate_predictions(request, project_id):
    """Triggers a Celery task to generate predictions."""
    try:
        data = json.loads(request.body)
        num_predictions = data.get('num_predictions', 10)
        
        task = generate_predictions_task.delay(project_id, num_predictions=num_predictions)
        
        message = "Prediction generation has been initiated. You will be notified upon completion."
        return JsonResponse({'status': 'ok', 'message': message, 'task_id': task.id})
    
    except Exception as e:
        return JsonResponse({'status': 'error', 'message': f'An unexpected error occurred: {e}'}, status=500)

@login_required
def get_predictions_data(request, project_id):
    """
    Fetches actual and predicted data for charting.
    Now handles both grouped and non-grouped (legacy) predictions.
    """
    project = get_object_or_404(Project, id=project_id)
    
    predictions_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), '..', 'trained')
    file_path = os.path.join(predictions_dir, f"project_{project.id}_latest_predictions.json")
    
    import json
    try:
        with open(file_path, 'r') as f:
            predictions_data = json.load(f)
    except FileNotFoundError:
        predictions_data = {}

    all_timestamps = set()
    grouped_predictions_data = {}

    # Check if it's the new format (grouped) or old format
    if predictions_data and 'labels' not in predictions_data and all(isinstance(v, dict) for v in predictions_data.values()):
        # New grouped format: dict of dicts
        grouped_predictions_data = predictions_data
        for group_name, predictions in grouped_predictions_data.items():
            all_timestamps.update(predictions.get('labels', []))
    elif predictions_data and 'labels' in predictions_data:
        # Old format: dict with 'labels' and 'predicted_data'
        grouped_predictions_data = {'default': predictions_data}
        all_timestamps.update(predictions_data.get('labels', []))

    actual_data = _get_actual_data_for_chart(project.id)
    
    # Process actual data for timestamps
    if isinstance(actual_data, dict):
        for group_name, group_data in actual_data.items():
            for col, data_points in group_data.items():
                all_timestamps.update([d['x'] for d in data_points])

    return JsonResponse({
        'labels': sorted(list(all_timestamps)),
        'actual_data': dict(actual_data),
        'predicted_data': grouped_predictions_data,
    })


@login_required
def download_predictions_csv(request, project_id):
    """
    Generates and serves a CSV file of the latest predictions for a project.
    """
    project = get_object_or_404(Project, id=project_id)
    
    predictions_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), '..', 'trained')
    file_path = os.path.join(predictions_dir, f"project_{project.id}_latest_predictions.json")

    try:
        with open(file_path, 'r') as f:
            predictions_data = json.load(f)
    except FileNotFoundError:
        # Or handle this with an error message
        return HttpResponse("No prediction data found.", status=404)

    # Convert the nested JSON to a flat list of dictionaries
    flat_data = []
    for group_name, group_data in predictions_data.items():
        # Timestamps are the labels for this group
        timestamps = group_data.get('labels', [])
        
        # This will hold the data for each timestamp in this group
        # { 'timestamp1': {'col1': val, 'col2': val}, 'timestamp2': ... }
        data_by_ts = {}

        # The actual predicted data is nested
        predicted_data_cols = group_data.get('predicted_data', {})
        
        for col_name, data_points in predicted_data_cols.items():
            for point in data_points:
                ts = point['x']
                if ts not in data_by_ts:
                    data_by_ts[ts] = {}
                data_by_ts[ts][f'predicted_{col_name}'] = point['y']
                # Add extra fields
                for key, value in point.items():
                    if key not in ['x', 'y']:
                        data_by_ts[ts][key] = value

        for ts in timestamps:
            row_data = {'group': group_name, 'timestamp': ts}
            row_data.update(data_by_ts.get(ts, {}))
            flat_data.append(row_data)

    if not flat_data:
        return HttpResponse("Prediction data is empty or in an unrecognized format.", status=404)

    df = pd.DataFrame(flat_data)
    
    # Reorder columns to be more intuitive
    cols = df.columns.tolist()
    if 'group' in cols:
        cols.insert(0, cols.pop(cols.index('group')))
    if 'timestamp' in cols:
        cols.insert(1, cols.pop(cols.index('timestamp')))
    df = df[cols]
    
    response = HttpResponse(content_type='text/csv')
    response['Content-Disposition'] = f'attachment; filename="project_{project.id}_predictions.csv"'

    df.to_csv(path_or_buf=response, index=False)
    
    return response

@login_required
def project_wizard_step3(request):
    db_data = request.session.get('wizard_db_data')
    selected_tables = request.session.get('wizard_selected_tables')

    if not db_data or not selected_tables:
        return redirect(reverse('wizard_step1'))

    if request.method == 'POST':
        form = ProjectForm(request.POST)
        if form.is_valid():
            try:
                with transaction.atomic():
                    # Create the Project
                    project = form.save()

                    # Create the ExternalDatabase connection
                    ExternalDatabase.objects.create(
                        project=project,
                        db_type=db_data['db_type'],
                        host=db_data['host'],
                        port=db_data['port'],
                        dbname=db_data['dbname'],
                        user=db_data['user'],
                        password=db_data['password']
                    )

                    # Create the SelectedTable entries
                    for table_name in selected_tables:
                        SelectedTable.objects.create(project=project, table_name=table_name)
                
                # Clean up session
                del request.session['wizard_db_data']
                del request.session['wizard_selected_tables']

                # TODO: Add success message
                return redirect(reverse('home'))
            except Exception as e:
                # TODO: Handle transaction error, maybe show an error message
                form.add_error(None, f"An error occurred while saving the project: {e}")

    else:
        form = ProjectForm()

    context = {
        'form': form,
        'hint': HINTS.get('project_wizard_step3')
    }
    return render(request, 'project_wizard_step3.html', context)

@login_required
@require_POST
def delete_project(request, project_id):
    """
    Deletes a project from the database.
    """
    project = get_object_or_404(Project, id=project_id)
    project.delete()
    return redirect(reverse('home'))