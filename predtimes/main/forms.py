from django import forms
from .models import ExternalDatabase, Project

class ProjectForm(forms.ModelForm):
    class Meta:
        model = Project
        fields = ['name', 'creator']

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.fields['name'].widget.attrs['class'] = 'form-control'
        self.fields['creator'].widget.attrs['class'] = 'form-control'

class ExternalDatabaseForm(forms.ModelForm):
    class Meta:
        model = ExternalDatabase
        fields = ['db_type', 'host', 'port', 'dbname', 'user', 'password']
        widgets = {
            'password': forms.PasswordInput(render_value=True),
        }
        labels = {
            'dbname': 'Database Name',
        }

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        for field_name, field in self.fields.items():
            field.widget.attrs['class'] = 'form-control'
            if field_name == 'db_type':
                field.widget.attrs['class'] += ' form-select'
