from django.urls import path
from . import views

urlpatterns = [
    path('', views.dashboard_view, name='dashboard'),
    path('generate-sample-predictions/', views.generate_sample_predictions_view, name='generate_predictions'),
    path('api/current-stats/', views.current_stats_api, name='current_stats_api'),
    path('api/sensor-data/', views.sensor_data_api, name='sensor_data_api'),
    path('api/debug-stats/', views.debug_current_stats, name='debug_stats'),
]