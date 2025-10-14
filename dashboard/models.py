from django.db import models
from django.utils import timezone

class SensorData(models.Model):
    temperature = models.FloatField(null=True, blank=True)
    humidity = models.FloatField(null=True, blank=True)
    motion = models.IntegerField(null=True, blank=True)
    battery_voltage = models.FloatField(null=True, blank=True)
    heat_index = models.FloatField(null=True, blank=True)
    dew_point = models.FloatField(null=True, blank=True)
    battery_percentage = models.FloatField(null=True, blank=True)
    temp_alert = models.BooleanField(default=False)
    timestamp = models.DateTimeField(default=timezone.now)
    
    class Meta:
        ordering = ['-timestamp']
        indexes = [
            models.Index(fields=['timestamp']),
            models.Index(fields=['temperature']),
        ]
    
    def __str__(self):
        return f"Sensor Data {self.timestamp}"

class Prediction(models.Model):
    PREDICTION_TYPES = [
        ('temperature', 'Temperature'),
        ('humidity', 'Humidity'),
    ]
    
    prediction_type = models.CharField(max_length=20, choices=PREDICTION_TYPES)
    predicted_value = models.FloatField()
    confidence = models.FloatField(null=True, blank=True)
    prediction_for = models.DateTimeField()  # Timestamp for which prediction is made
    created_at = models.DateTimeField(auto_now_add=True)
    
    class Meta:
        ordering = ['-created_at']
    
    def __str__(self):
        return f"{self.prediction_type} prediction for {self.prediction_for}"