"""
URL configuration for the main application.

This module defines the URL patterns for the main app, mapping URLs to their
corresponding view functions.
"""
from django.urls import path
from . import views

urlpatterns = [
    path('login/', views.login_view, name='login'),
    path('logout/', views.logout_view, name='logout'),
    path('', views.home_view, name='home'),
    
    # Wizard URLs
    path('project/wizard/step1/', views.project_wizard_step1, name='wizard_step1'),
    path('project/wizard/step2/', views.project_wizard_step2, name='wizard_step2'),
    path('project/wizard/step3/', views.project_wizard_step3, name='wizard_step3'),
    path('project/wizard/test_connection/', views.test_db_connection, name='test_db_connection'),

    # Project URLs
    path('project/<uuid:project_id>/', views.project_detail, name='project_detail'),
    path('project/<uuid:project_id>/save_column_selection/', views.save_column_selection, name='save_column_selection'),
    path('project/<uuid:project_id>/start_training/', views.start_training, name='start_training'),
    path('training_session/<int:training_session_id>/status/', views.get_training_status, name='get_training_status'),
    path('project/<uuid:project_id>/generate_predictions/', views.generate_predictions, name='generate_predictions'),
    path('task/<str:task_id>/status/', views.get_prediction_status, name='get_prediction_status'),
    path('project/<uuid:project_id>/get_predictions_data/', views.get_predictions_data, name='get_predictions_data'),
    path('project/<uuid:project_id>/download_predictions_csv/', views.download_predictions_csv, name='download_predictions_csv'),
    path('project/<uuid:project_id>/delete/', views.delete_project, name='delete_project'),
    path('task/<str:task_id>/get_table_data/', views.get_table_data, name='get_table_data'),
    path('project/<uuid:project_id>/trigger_table_data_refresh/', views.trigger_table_data_refresh, name='trigger_table_data_refresh'),
]
