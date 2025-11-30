from sqlalchemy import create_engine
from .models import Project

def get_db_engine(project_id):
    """
    Creates and returns a SQLAlchemy engine for a given project.
    """
    project = Project.objects.get(id=project_id)
    db_details = project.database

    dialect_map = {
        'postgresql': 'postgresql+psycopg2',
        'mysql': 'mysql+pymysql',
        'sqlserver': 'mssql+pyodbc',
        'oracle': 'oracle+cx_oracle',
        'sqlite': 'sqlite',
    }
    dialect = dialect_map.get(db_details.db_type)

    if not dialect:
        raise ValueError(f"Unsupported database type: {db_details.db_type}")

    if db_details.db_type == 'sqlite':
        # For SQLite, the 'host' field stores the full file path
        url = f'{dialect}:///{db_details.host}'
    else:
        url = f'{dialect}://{db_details.user}:{db_details.password}@{db_details.host}:{db_details.port}/{db_details.dbname}'
    
    return create_engine(url)

def get_db_engine_from_details(db_type, host, port, dbname, user, password):
    """
    Creates and returns a SQLAlchemy engine from raw connection details.
    """
    dialect_map = {
        'postgresql': 'postgresql+psycopg2',
        'mysql': 'mysql+pymysql',
        'sqlserver': 'mssql+pyodbc',
        'oracle': 'oracle+cx_oracle',
        'sqlite': 'sqlite',
    }
    dialect = dialect_map.get(db_type)

    if not dialect:
        raise ValueError(f"Unsupported database type: {db_type}")

    if db_type == 'sqlite':
        url = f'{dialect}:///{host}'
    else:
        url = f'{dialect}://{user}:{password}@{host}:{port}/{dbname}'
    
    return create_engine(url)
