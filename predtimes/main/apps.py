"""
This file contains the application configuration for the main Django app.
"""
from django.apps import AppConfig


class MainConfig(AppConfig):
    """
    Configuration for the 'main' application.
    """
    default_auto_field = 'django.db.models.BigAutoField'
    name = 'main'
