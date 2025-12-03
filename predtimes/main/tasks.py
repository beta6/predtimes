from celery import shared_task, current_task
from datetime import timedelta
import os
import numpy as np
from sqlalchemy import create_engine, text, select, table, column
from sklearn.preprocessing import MinMaxScaler
from .models import Project, ProjectColumn, ExternalDatabase, SelectedTable
from .ml_models import get_model
from .db_utils import get_db_engine
import pandas as pd
from .somecalls import _get_training_data_from_db, save_table_data
import joblib

def create_patches(time_series, patch_len, stride):
    """
    Creates patches from a time series.

    This is used to prepare data for the PatchTST model.

    Args:
        time_series: The input time series data.
        patch_len: The length of each patch.
        stride: The stride between patches.

    Returns:
        A numpy array of patches.
    """
    patches = []
    for i in range(0, len(time_series) - patch_len + 1, stride):
        patches.append(time_series[i : i + patch_len])
    return np.array(patches)

@shared_task(bind=True)
def train_model_task(self, training_session_id):
    """
    Celery task to train a time series prediction model for each group in the data.
    """
    from .models import TrainingSession, Model
    from datetime import datetime

    training_session = TrainingSession.objects.get(id=training_session_id)
    project = training_session.project
    architecture = training_session.model_architecture

    training_session.status = 'TRAINING'
    training_session.details = 'Starting training process...'
    training_session.save()
    print(f"Starting training for session {training_session_id}...")

    try:
        # 1. Fetch column info
        datetime_cols = ProjectColumn.objects.filter(project=project, column_type='datetime')
        numeric_cols = ProjectColumn.objects.filter(project=project, column_type='numeric')
        multigroup_cols = ProjectColumn.objects.filter(project=project, column_type='multigroup')

        if not datetime_cols or not numeric_cols:
            raise ValueError("Datetime and numeric columns must be selected before training.")

        # Store group columns in the session for later reference
        training_session.group_columns = [c.column_name for c in multigroup_cols]
        training_session.save()

        # 2. Fetch data and save to CSV
        data_filepath = save_table_data(project.id, datetime_cols, numeric_cols, multigroup_cols)
        training_session.details = 'Data preparation complete. Starting training for each group...'
        training_session.save()

        # 3. Load data from the CSV file using pandas
        df = pd.read_csv(data_filepath)
        print("Data loaded from CSV.\n")

        dt_col_name = datetime_cols[0].column_name
        numeric_col_names = [c.column_name for c in numeric_cols]
        multigroup_col_names = [c.column_name for c in multigroup_cols]

        # Convert date column to datetime
        df[dt_col_name] = pd.to_datetime(df[dt_col_name])

        # Use only the numeric columns that actually exist in the dataframe
        available_numeric_cols = [col for col in numeric_col_names if col in df.columns]
        
        if not available_numeric_cols:
            raise ValueError("None of the specified numeric columns were found in the data.")
        
        # Ensure correct dtype for available numeric columns
        for col in available_numeric_cols:
            df[col] = pd.to_numeric(df[col], errors='coerce').fillna(0).round().astype(int)
        
        numeric_col_names = available_numeric_cols

        if not multigroup_col_names:
            # No multigroup columns, train a single model
            groups = [None]
        else:
            # Create a composite key for grouping
            df['group_key'] = df[multigroup_col_names].astype(str).agg('_'.join, axis=1)
            groups = df['group_key'].unique()

        num_models_created = 0
        for group in groups:
            if group is None:
                group_df = df
                group_name = 'default'
                training_session.details = f'Training model for default group...'
            else:
                group_df = df[df['group_key'] == group]
                group_name = group
                training_session.details = f'Training model for group: {group_name}...'
            training_session.save()

            # Sort data for the group
            group_df = group_df.sort_values(by=dt_col_name)

            # Filter numeric columns for the current group
            group_numeric_cols = [col for col in numeric_col_names if col in group_df.columns]

            if not group_numeric_cols:
                print(f"Skipping group {group_name}: Not enough data for training.")
                continue

            # Normalize the numeric data for the group
            numeric_data = group_df[group_numeric_cols].values
            scaler = MinMaxScaler(feature_range=(0, 1))
            scaled_data = scaler.fit_transform(numeric_data)

            # Create time series sequences
            sequence_length = 10
            if len(scaled_data) <= sequence_length:
                print(f"Skipping group {group_name}: Not enough data for training.")
                continue
            
            X, y = [], []
            num_features = len(group_numeric_cols)

            if architecture == 'patchtst':
                patch_len = 16
                stride = 8
                if len(scaled_data) <= patch_len:
                    print(f"Skipping group {group_name}: Not enough data for training.")
                    continue
                X = create_patches(scaled_data[:-1], patch_len, stride)
                y = scaled_data[patch_len:]

                if X.shape[0] > y.shape[0]:
                    X = X[:y.shape[0]]
                elif y.shape[0] > X.shape[0]:
                    y = y[:X.shape[0]]

            else:
                for i in range(len(scaled_data) - sequence_length):
                    X.append(scaled_data[i:(i + sequence_length), :])
                    y.append(scaled_data[i + sequence_length, :])
            
            X, y = np.array(X), np.array(y)

            if architecture != 'patchtst':
                X = np.reshape(X, (X.shape[0], X.shape[1], num_features))

            # Build and compile the model
            input_shape = (X.shape[1], X.shape[2]) if len(X.shape) > 2 else (X.shape[1],)
            model = get_model(architecture, input_shape, num_features)

            model.compile(optimizer='adam', loss='mse', metrics=['mse'])
            
            # Train the model
            print(f"Fitting model for group {group_name}...")
            model.fit(X, y, epochs=10, batch_size=32, verbose=0) # verbose=0 to reduce log spam

            # Save the trained model and scaler
            trained_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), '..', 'trained')
            os.makedirs(trained_dir, exist_ok=True)
            
            base_filename = f"session_{training_session.id}_group_{group_name}"
            model_filename = f"{base_filename}_model.h5"
            scaler_filename = f"{base_filename}_scaler.joblib"

            model_path = os.path.join(trained_dir, model_filename)
            scaler_path = os.path.join(trained_dir, scaler_filename)
            
            model.save(model_path)
            joblib.dump(scaler, scaler_path)

            # Save model info to Django DB
            Model.objects.create(
                project=project,
                architecture=architecture,
                model_path=model_path,
                scaler_path=scaler_path,
                group_name=group_name,
                numeric_columns=group_numeric_cols,
                training_session=training_session
            )
            num_models_created += 1

        if num_models_created == 0:
            raise ValueError("Training completed, but no models were created. This is likely due to insufficient data in all groups.")

        training_session.status = 'SUCCESS'
        training_session.details = "Training complete for all groups."
        training_session.save()

        print(f"Training complete for session {training_session_id}.")
        return {'status': 'SUCCESS', 'message': f"Training complete for project {project.id}."}

    except Exception as e:
        print(f"An error occurred during training for session {training_session_id}: {e}")
        training_session.status = 'FAILURE'
        training_session.details = f'Training failed: {str(e)}'
        training_session.save()
        raise e

@shared_task(bind=True)
def generate_predictions_task(self, project_id, num_predictions=10):
    """
    Celery task to generate future predictions for each group in a project.

    This task loads the latest trained models for a project, fetches the most
    recent data from the external database, and generates a specified number of
    future predictions. The predictions are saved to a JSON file.

    Args:
        project_id: The ID of the project to generate predictions for.
        num_predictions: The number of future predictions to generate.

    Returns:
        A success message if predictions are generated successfully.
    """
    from .models import TrainingSession
    from datetime import datetime
    from tensorflow.keras.models import load_model
    from collections import defaultdict
    import json

    print(f"Starting prediction generation for project {project_id}...")

    try:
        project = Project.objects.get(id=project_id)
        latest_training_session = project.training_sessions.latest('created_at')
        models_to_predict = latest_training_session.models.all()
        architecture = latest_training_session.model_architecture

        if not models_to_predict:
            raise ValueError("No models found for the latest training session.")

        datetime_cols = ProjectColumn.objects.filter(project=project, column_type='datetime')
        numeric_cols = ProjectColumn.objects.filter(project=project, column_type='numeric')
        multigroup_cols = ProjectColumn.objects.filter(project=project, column_type='multigroup')

        if not datetime_cols.exists():
            raise ValueError("No datetime column selected for prediction.")

        dt_col_info = datetime_cols.first()
        numeric_col_info = list(numeric_cols)
        multigroup_col_info = list(multigroup_cols)
        
        engine = get_db_engine(project.id)
        table_name = dt_col_info.table_name
        db_table = table(table_name)
        dt_col = column(dt_col_info.column_name)

        all_predictions = {}

        for model_record in models_to_predict:
            model = load_model(model_record.model_path, compile=False)
            scaler = joblib.load(model_record.scaler_path)
            sequence_length = model.input_shape[1]
            group_name = model_record.group_name
            if group_name is None:
                group_name = 'default'

            # Get numeric columns used for this specific model
            numeric_col_names = model_record.numeric_columns
            if not numeric_col_names:
                # Fallback for older models
                numeric_col_names = [c.column_name for c in numeric_col_info]


            # Fetch data for the specific group, including all datetime columns
            datetime_col_names = [c.column_name for c in datetime_cols]
            
            cols_to_select = [column(c) for c in datetime_col_names] + [column(c) for c in numeric_col_names]
            
            query = select(*cols_to_select).select_from(db_table)

            if group_name != 'default':
                group_filters = []
                group_values = group_name.split('_')
                for i, col_info in enumerate(multigroup_col_info):
                    group_filters.append(column(col_info.column_name) == group_values[i])
                query = query.where(*group_filters)

            if architecture == 'patchtst':
                patch_len = 16
                query = query.order_by(dt_col.desc()).limit(patch_len)
            else:
                query = query.order_by(dt_col.desc()).limit(sequence_length)

            with engine.connect() as connection:
                result = connection.execute(query)
                headers = list(result.keys())
                rows = result.fetchall()
            
            if architecture == 'patchtst':
                if len(rows) < patch_len:
                    print(f"Skipping group {group_name}: Not enough data for prediction.")
                    continue
            else:
                if len(rows) < sequence_length:
                    print(f"Skipping group {group_name}: Not enough data for prediction.")
                    continue

            rows.reverse()
            
            last_row_dict = dict(zip(headers, rows[-1]))
            last_timestamp = pd.to_datetime(last_row_dict[dt_col_info.column_name])

            # Extract extra datetime fields from the last known row
            extra_dt_fields = {}
            for col_name in datetime_col_names:
                if col_name != dt_col_info.column_name:
                    value = last_row_dict.get(col_name)
                    if hasattr(value, 'isoformat'):
                        extra_dt_fields[col_name] = value.isoformat()
                    else:
                        extra_dt_fields[col_name] = str(value) if value is not None else None

            numeric_indices = [headers.index(name) for name in numeric_col_names]
            
            numeric_data = [[row[i] for i in numeric_indices] for row in rows]
            numeric_data = np.array(numeric_data, dtype=float)
            numeric_data = np.round(numeric_data).astype(int)
            scaled_data = scaler.transform(numeric_data)
            
            # Generate predictions
            predictions = []
            num_features = len(numeric_col_names)

            if architecture == 'patchtst':
                current_sequence = scaled_data.reshape(1, scaled_data.shape[0], scaled_data.shape[1])
            else:
                current_sequence = scaled_data.reshape(1, sequence_length, num_features)

            for _ in range(num_predictions):
                next_pred_scaled = model.predict(current_sequence)

                if architecture == 'dlinear':
                    # For sequence-predicting models like DLinear, we do recursive forecasting one step at a time.
                    # The model outputs a whole sequence, so we take just the first predicted step.
                    prediction = next_pred_scaled[:, 0, :].reshape(1, num_features)
                else:
                    # For single-step models, the output is already the prediction.
                    prediction = next_pred_scaled

                predictions.append(prediction[0])

                # Roll the sequence forward by removing the oldest step and adding the new prediction.
                new_sequence_base = current_sequence[0, 1:, :]
                new_sequence = np.vstack([new_sequence_base, prediction])
                
                if architecture == 'patchtst':
                    current_sequence = new_sequence.reshape(1, new_sequence.shape[0], new_sequence.shape[1])
                else:
                    current_sequence = new_sequence.reshape(1, sequence_length, num_features)


            predicted_values = scaler.inverse_transform(np.array(predictions))
            predicted_values = np.round(predicted_values).astype(int)
            
            time_deltas = pd.to_datetime(pd.Series([r[headers.index(dt_col.name)] for r in rows])).diff().median()

            group_predictions = defaultdict(list)
            group_timestamps = []

            for i in range(num_predictions):
                pred_timestamp = last_timestamp + time_deltas * (i + 1)
                ts_iso = pred_timestamp.isoformat()
                group_timestamps.append(ts_iso)
                for j, col_name in enumerate(numeric_col_names):
                    pred_value = predicted_values[i, j]
                    
                    point_data = {
                        'x': ts_iso,
                        'y': int(pred_value),
                        **extra_dt_fields
                    }
                    group_predictions[col_name].append(point_data)
            
            all_predictions[group_name] = {
                'labels': group_timestamps,
                'predicted_data': group_predictions
            }

        # Save all predictions to a single file
        predictions_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), '..', 'trained')
        os.makedirs(predictions_dir, exist_ok=True)
        file_path = os.path.join(predictions_dir, f"project_{project.id}_latest_predictions.json")
        
        with open(file_path, 'w') as f:
            json.dump(all_predictions, f)

        print(f"Prediction generation complete for project {project_id}. Saved to {file_path}")
        return f"Generated {num_predictions} predictions for project {project_id}."

    except Exception as e:
        print(f"An error occurred during prediction for project {project_id}: {e}")
        raise e

def _get_table_sample_data(project_id):
    """
    Helper function to fetch a small sample of data from the external tables
    associated with a project.

    This function connects to the project's external database and retrieves the
    first 5 rows from each selected table. This data is used to provide a
    preview to the user in the UI.

    Args:
        project_id: The ID of the project to fetch data for.

    Returns:
        A dictionary containing the table data and any error messages.
    """
    from sqlalchemy.exc import SQLAlchemyError

    project = Project.objects.get(id=project_id)
    table_data = {}
    error_message = None

    try:
        engine = get_db_engine(project.id)
        with engine.connect() as connection:
            for selected_table in project.selected_tables.all():
                table_name = selected_table.table_name
                query = text(f"SELECT * FROM {table_name} LIMIT 5")
                
                result = connection.execute(query)
                headers = list(result.keys()) # Convert to list for JSON serialization
                
                processed_rows = []
                for row in result.fetchall():
                    processed_row = []
                    for item in row:
                        if isinstance(item, timedelta):
                            processed_row.append(item.total_seconds())
                        else:
                            processed_row.append(item)
                    processed_rows.append(processed_row)
                table_data[table_name] = {'headers': headers, 'rows': processed_rows}

    except (ImportError, SQLAlchemyError, ValueError) as e:
        error_message = f"Could not load data from external database: {e}. Ensure required libraries are installed and connection details are correct."
    except Exception as e:
        error_message = f"An unexpected error occurred: {e}"
    
    return {'table_data': table_data, 'error_message': error_message}

@shared_task
def get_table_sample_data_task(project_id):
    """
    Celery task to fetch a sample of data for a project's selected tables.

    This task is intended to be called asynchronously to avoid blocking the main
    application thread while fetching data from the external database.

    Args:
        project_id: The ID of the project to fetch data for.

    Returns:
        The result of the _get_table_sample_data function.
    """
    return _get_table_sample_data(project_id)

