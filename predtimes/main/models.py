from django.db import models
import uuid

class Project(models.Model):
    id = models.UUIDField(primary_key=True, default=uuid.uuid4, editable=False)
    name = models.CharField(max_length=255)
    creator = models.CharField(max_length=255)
    created_at = models.DateTimeField(auto_now_add=True)

    def __str__(self):
        return self.name

class ExternalDatabase(models.Model):
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
    project = models.ForeignKey(Project, on_delete=models.CASCADE, related_name='selected_tables')
    table_name = models.CharField(max_length=255)

    class Meta:
        unique_together = ('project', 'table_name')

    def __str__(self):
        return self.table_name


class ProjectColumn(models.Model):
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

class Prediction(models.Model):
    project = models.ForeignKey(Project, on_delete=models.CASCADE, related_name='predictions')
    model = models.ForeignKey(Model, on_delete=models.CASCADE)
    timestamp = models.DateTimeField()
    value = models.FloatField()
    created_at = models.DateTimeField(auto_now_add=True)

    class Meta:
        ordering = ['timestamp']

    def __str__(self):
        return f"Prediction for {self.project.name} at {self.timestamp}: {self.value}"


class TrainingSession(models.Model):
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
    model = models.ForeignKey(Model, on_delete=models.SET_NULL, null=True, blank=True, related_name='training_sessions_as_model')
    status = models.CharField(max_length=20, choices=STATUS_CHOICES, default='PENDING')
    details = models.TextField(blank=True, null=True)
    group_columns = models.JSONField(null=True, blank=True)
    created_at = models.DateTimeField(auto_now_add=True)
    updated_at = models.DateTimeField(auto_now=True)

    def __str__(self):
        return f"Training for {self.project.name} ({self.model_architecture}) - {self.status}"

class TrainingData(models.Model):
    session = models.ForeignKey(TrainingSession, on_delete=models.CASCADE, related_name='data_dump')
    data = models.JSONField()
    timestamp = models.DateTimeField(auto_now_add=True)

    class Meta:
        ordering = ['timestamp']