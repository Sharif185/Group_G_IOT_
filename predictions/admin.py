from django.contrib import admin
from .models import Prediction

@admin.register(Prediction)
class PredictionAdmin(admin.ModelAdmin):
    list_display = ['prediction_type', 'predicted_value', 'prediction_for', 'created_at']
    list_filter = ['prediction_type', 'created_at']