from django.contrib import admin
from .models import SensorData

@admin.register(SensorData)
class SensorDataAdmin(admin.ModelAdmin):
    list_display = ['timestamp', 'temperature', 'humidity', 'battery_voltage', 'temp_alert']
    list_filter = ['timestamp', 'temp_alert']
    readonly_fields = ['timestamp']
    date_hierarchy = 'timestamp'