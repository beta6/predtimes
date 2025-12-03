"""
Custom template tags and filters for the PredTimes application.
"""
from django import template
import json
from django.utils.safestring import mark_safe

register = template.Library()

@register.filter(is_safe=True)
def safe_json(data):
    """
    Safely JSON-encode an object.
    """
    return mark_safe(json.dumps(data))

@register.filter
def get_item(dictionary, key):
    """Allows accessing dictionary items with a variable key in templates."""
    return dictionary.get(key)


@register.filter
def zip_lists(a, b):
    """Zips two lists together."""
    return zip(a, b)

@register.simple_tag
def project_columns_by_table(project, table_name):
    """Returns a dictionary of column types for a specific table."""
    columns = project.columns.filter(table_name=table_name)
    return {col.column_name: col.column_type for col in columns}

@register.simple_tag
def get_column_class(header, columns):
    """Returns the CSS class for a given header based on the columns dict."""
    column_type = columns.get(header)
    if column_type == 'datetime':
        return 'datetime-column'
    elif column_type == 'numeric':
        return 'numeric-column'
    return ''
