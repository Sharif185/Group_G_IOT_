from django.db import models
from django.utils import timezone

class Prediction(models.Model):
    PREDICTION_TYPES = [
        ('temperature', 'Temperature'),
        ('humidity', 'Humidity'),
    ]
    
    prediction_type = models.CharField(max_length=20, choices=PREDICTION_TYPES)
    predicted_value = models.FloatField()
    confidence = models.FloatField()  # 0.0 to 1.0
    prediction_for = models.DateTimeField()  # When this prediction is for
    created_at = models.DateTimeField(auto_now_add=True)
    
    class Meta:
        ordering = ['prediction_for', 'prediction_type']
    
    def __str__(self):
        return f"{self.prediction_type} prediction for {self.prediction_for}"