from sqlalchemy import select, table, column, text
from .models import Project
from .db_utils import get_db_engine
from datetime import timedelta, datetime
from celery import shared_task
import os
import pandas as pd

def _get_training_data_from_db(project_id, datetime_cols, numeric_cols, multigroup_cols):
    """
    Helper function to fetch training data (all datetime and numeric columns)
    from the external DB for a given project.
    Returns a pandas DataFrame.
    """
    project = Project.objects.get(id=project_id)
    
    if not datetime_cols or not numeric_cols:
        raise ValueError("Datetime and numeric columns must be selected before training.")

    dt_col_info = datetime_cols[0]
    numeric_col_info = list(numeric_cols)
    multigroup_col_info = list(multigroup_cols)

    engine = get_db_engine(project.id)
    
    table_name = dt_col_info.table_name
    
    db_table = table(table_name)
    dt_col = column(dt_col_info.column_name)
    
    query = select(text('*')).select_from(db_table)
    
    with engine.connect() as connection:
        result = connection.execute(query)
        rows = result.fetchall()
        columns = result.keys()
        df = pd.DataFrame(rows, columns=columns)
    
    return df

def save_table_data(project_id, datetime_cols, numeric_cols, multigroup_cols):
    """
    Fetches data from the project's database and saves it to a temporary CSV file.

    This function is used to create a snapshot of the training data, which can
    be useful for debugging or for training models outside of the main application.

    Args:
        project_id: The ID of the project to fetch data for.
        datetime_cols: A list of the datetime columns.
        numeric_cols: A list of the numeric columns.
        multigroup_cols: A list of the multigroup columns.

    Returns:
        The path to the saved CSV file.
    """
    df = _get_training_data_from_db(project_id, datetime_cols, numeric_cols, multigroup_cols)
    
    # Ensure temp_data directory exists
    temp_data_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), '..', 'temp_data')
    os.makedirs(temp_data_dir, exist_ok=True)
    
    file_path = os.path.join(temp_data_dir, f"project_{project_id}_training_data.csv")
    
    df.to_csv(file_path, index=False)
    
    return file_path


@shared_task
def _get_actual_data_for_chart(project_id, num_days_to_fetch=365):
    """
    Asynchronous Celery task to fetch recent actual data from the external DB for
    charting purposes.

    This function retrieves the most recent data points from the user's database
    to be displayed on the project's detail page. It groups the data by the
    configured 'multigroup' columns.

    Args:
        project_id: The ID of the project to fetch data for.
        num_days_to_fetch: The number of days of recent data to fetch.

    Returns:
        A dictionary containing the actual data, grouped by group key.
    """
    from collections import defaultdict
    actual_data_by_group = defaultdict(lambda: defaultdict(list))
    try:
        project = Project.objects.get(id=project_id)
        dt_col_info = project.columns.filter(column_type='datetime').first()
        numeric_cols = project.columns.filter(column_type='numeric')
        multigroup_cols = project.columns.filter(column_type='multigroup')

        if dt_col_info and numeric_cols.exists():
            engine = get_db_engine(project.id)
            
            table_name = dt_col_info.table_name
            db_table = table(table_name)
            dt_col = column(dt_col_info.column_name)
            
            cols_to_select = [dt_col] + [column(c.column_name) for c in numeric_cols] + [column(c.column_name) for c in multigroup_cols]
            
            query = select(*cols_to_select).select_from(db_table).order_by(dt_col.desc()).limit(num_days_to_fetch * (len(multigroup_cols) or 1))
            
            with engine.connect() as connection:
                result = connection.execute(query)
                headers = result.keys()
                rows = result.fetchall()
            
            df = pd.DataFrame(rows, columns=headers)
            
            if not multigroup_cols:
                df['group_key'] = 'default'
            else:
                multigroup_col_names = [c.column_name for c in multigroup_cols]
                df['group_key'] = df[multigroup_col_names].astype(str).agg('_'.join, axis=1)

            df[dt_col_info.column_name] = pd.to_datetime(df[dt_col_info.column_name])
            df = df.sort_values(by=dt_col_info.column_name)

            for group_name, group_df in df.groupby('group_key'):
                for _, row in group_df.iterrows():
                    timestamp = row[dt_col_info.column_name]
                    ts_iso = timestamp.isoformat()
                    
                    for num_col in numeric_cols:
                        value = row[num_col.column_name]
                        if isinstance(value, timedelta):
                            value = value.total_seconds()
                        
                        actual_data_by_group[group_name][num_col.column_name].append({'x': ts_iso, 'y': value})
    
    except Exception as e:
        print(f"Could not fetch actual data for chart: {e}")
        
    return actual_data_by_group
