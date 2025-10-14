from django.apps import AppConfig

class DashboardConfig(AppConfig):
    default_auto_field = 'django.db.models.BigAutoField'
    name = 'dashboard'
    
    def ready(self):
        # Start MQTT service when Django starts
        try:
            from .mqtt_service import mqtt_service
            mqtt_service.start()
            print("✅ MQTT service started")
        except Exception as e:
            print(f"❌ Error starting MQTT service: {e}")