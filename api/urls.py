from django.urls import path
from dashboard import views as dashboard_views

urlpatterns = [
    path('sensor-data/', dashboard_views.sensor_data_api, name='sensor_data_api'),
    path('current-stats/', dashboard_views.current_stats_api, name='current_stats_api'),
]