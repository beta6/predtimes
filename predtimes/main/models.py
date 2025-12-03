from django.db import models
import uuid

class Project(models.Model):
    """
    Represents a single time series prediction project.

    Each project is a container for database connections, selected tables,
    trained models, and predictions.

    Attributes:
        id: A unique identifier for the project.
        name: The name of the project.
        creator: The name of the user who created the project.
        created_at: The date and time the project was created.
    """
    id = models.UUIDField(primary_key=True, default=uuid.uuid4, editable=False)
    name = models.CharField(max_length=255)
    creator = models.CharField(max_length=255)
    created_at = models.DateTimeField(auto_now_add=True)

    def __str__(self):
        return self.name

class ExternalDatabase(models.Model):
    """
    Stores the connection details for an external database.

    Each project has one associated ExternalDatabase, which holds the
    credentials and other information needed to connect to the user's
    time series data source.

    Attributes:
        project: A one-to-one relationship to the associated Project.
        db_type: The type of the database (e.g., PostgreSQL, MySQL).
        host: The database host address.
        port: The database port number.
        dbname: The name of the database.
        user: The username for database authentication.
        password: The password for database authentication. This should be
                  encrypted in a production environment.
    """
    DATABASE_TYPE_CHOICES = [
        ('postgresql', 'PostgreSQL'),
        ('mysql', 'MySQL'),
        ('oracle', 'Oracle'),
        ('sqlserver', 'Microsoft SQL Server'),
        ('sqlite', 'SQLite'),
    ]

    project = models.OneToOneField(Project, on_delete=models.CASCADE, related_name='database')
    db_type = models.CharField(max_length=50, choices=DATABASE_TYPE_CHOICES)
    host = models.CharField(max_length=255)
    port = models.IntegerField()
    dbname = models.CharField(max_length=255, verbose_name="Database Name")
    user = models.CharField(max_length=255)
    password = models.CharField(max_length=255) # In a real app, this should be encrypted

    def __str__(self):
        return f"{self.get_db_type_display()} at {self.host}"

class SelectedTable(models.Model):
    """
    Represents a database table that has been selected for use in a project.

    Attributes:
        project: A foreign key to the associated Project.
        table_name: The name of the selected database table.
    """
    project = models.ForeignKey(Project, on_delete=models.CASCADE, related_name='selected_tables')
    table_name = models.CharField(max_length=255)

    class Meta:
        unique_together = ('project', 'table_name')

    def __str__(self):
        return self.table_name


class ProjectColumn(models.Model):
    """
    Represents a column from a selected table that has been configured for a
    specific role in the project.

    This model maps columns to their intended use, such as 'datetime', 'numeric'
    (the value to be predicted), or 'multigroup' (for grouping data).

    Attributes:
        project: A foreign key to the associated Project.
        table_name: The name of the table containing the column.
        column_name: The name of the column.
        column_type: The role of the column in the prediction task.
    """
    COLUMN_TYPE_CHOICES = [
        ('datetime', 'Datetime'),
        ('numeric', 'Numeric'),
        ('multigroup', 'Multigroup'),
    ]
    project = models.ForeignKey(Project, on_delete=models.CASCADE, related_name='columns')
    table_name = models.CharField(max_length=255)
    column_name = models.CharField(max_length=255)
    column_type = models.CharField(max_length=50, choices=COLUMN_TYPE_CHOICES)

    class Meta:
        unique_together = ('project', 'table_name', 'column_name')

    def __str__(self):
        return f"{self.table_name}.{self.column_name} ({self.get_column_type_display()})"

class Model(models.Model):
    """
    Represents a trained machine learning model.

    This model stores information about a trained model, including its
    architecture, file path, and associated training session.

    Attributes:
        project: A foreign key to the associated Project.
        architecture: The name of the model architecture (e.g., 'conv1d',
                      'transformer').
        model_path: The file path where the trained model is saved.
        scaler_path: The file path where the data scaler is saved.
        group_name: The name of the group if the model is for a specific
                    group of data.
        numeric_columns: A JSON field to store the list of numeric columns
                         used for training.
        training_session: A foreign key to the TrainingSession that produced
                          this model.
        created_at: The date and time the model was created.
    """
    project = models.ForeignKey(Project, on_delete=models.CASCADE, related_name='models')
    architecture = models.CharField(max_length=100)
    model_path = models.CharField(max_length=512)
    scaler_path = models.CharField(max_length=512)
    group_name = models.CharField(max_length=255, null=True, blank=True)
    numeric_columns = models.JSONField(null=True, blank=True)  # To store the list of numeric columns
    training_session = models.ForeignKey('TrainingSession', on_delete=models.SET_NULL, null=True, blank=True, related_name='models')
    created_at = models.DateTimeField(auto_now_add=True)

    def __str__(self):
        return f"Model for {self.project.name} ({self.architecture})"


class TrainingSession(models.Model):
    """
    Represents a single model training session.

    This model tracks the status and details of a Celery task that trains a
    machine learning model.

    Attributes:
        project: A foreign key to the associated Project.
        celery_task_id: The ID of the Celery task for this training session.
        model_architecture: The architecture of the model being trained.
        status: The current status of the training session (e.g., 'PENDING',
                'STARTED', 'SUCCESS', 'FAILURE').
        details: A text field for storing details or logs about the training
                 session.
        group_columns: A JSON field to store the group columns used for this
                       training session.
        created_at: The date and time the training session was created.
        updated_at: The date and time the training session was last updated.
    """
    STATUS_CHOICES = [
        ('PENDING', 'Pending'),
        ('STARTED', 'Started'),
        ('SUCCESS', 'Success'),
        ('FAILURE', 'Failure'),
        ('PROGRESS', 'In Progress'),
    ]
    project = models.ForeignKey(Project, on_delete=models.CASCADE, related_name='training_sessions')
    celery_task_id = models.CharField(max_length=255, unique=True, null=True, blank=True)
    model_architecture = models.CharField(max_length=100)
    status = models.CharField(max_length=20, choices=STATUS_CHOICES, default='PENDING')
    details = models.TextField(blank=True, null=True)
    group_columns = models.JSONField(null=True, blank=True)
    created_at = models.DateTimeField(auto_now_add=True)
    updated_at = models.DateTimeField(auto_now=True)

    def __str__(self):
        return f"Training for {self.project.name} ({self.model_architecture}) - {self.status}"

class TrainingData(models.Model):
    """
    Stores a snapshot of the data used for a training session.

    This is useful for debugging and for keeping a record of the exact data
    that was used to train a model.

    Attributes:
        session: A foreign key to the associated TrainingSession.
        data: A JSON field containing the training data.
        timestamp: The date and time the data was saved.
    """
    session = models.ForeignKey(TrainingSession, on_delete=models.CASCADE, related_name='data_dump')
    data = models.JSONField()
    timestamp = models.DateTimeField(auto_now_add=True)

    class Meta:
        ordering = ['timestamp']