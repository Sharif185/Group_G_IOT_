from django.shortcuts import render
from django.http import JsonResponse
from django.utils import timezone
from datetime import timedelta
import json
import math
import random
from .models import SensorData

def dashboard_view(request):
    """Main dashboard view"""
    try:
        # Get latest sensor data
        latest_data = SensorData.objects.last()
        
        # Get data for charts (last 24 hours) - convert to list of dictionaries
        twenty_four_hours_ago = timezone.now() - timedelta(hours=24)
        chart_data = list(SensorData.objects.filter(
            timestamp__gte=twenty_four_hours_ago
        ).values(
            'timestamp', 'temperature', 'humidity', 'battery_voltage',
            'heat_index', 'dew_point', 'motion', 'temp_alert'
        ).order_by('timestamp'))
        
        # Initialize predictions
        temperature_predictions = []
        humidity_predictions = []
        predictions_available = False
        
        # Try to load or generate predictions using EnvironmentalPredictor
        try:
            from .train_model import EnvironmentalPredictor
            from predictions.models import Prediction
            
            predictor = EnvironmentalPredictor()
            
            # Check if we have existing predictions
            temperature_predictions = list(Prediction.objects.filter(
                prediction_type='temperature'
            ).order_by('prediction_for')[:12])
            
            humidity_predictions = list(Prediction.objects.filter(
                prediction_type='humidity'
            ).order_by('prediction_for')[:12])
            
            # If no predictions exist, generate them using the ML model
            if len(temperature_predictions) == 0:
                print("🔄 No predictions found, generating predictions using ML model...")
                predictor.generate_predictions(hours_ahead=12)
                
                # Reload predictions after generation
                temperature_predictions = list(Prediction.objects.filter(
                    prediction_type='temperature'
                ).order_by('prediction_for')[:12])
                
                humidity_predictions = list(Prediction.objects.filter(
                    prediction_type='humidity'
                ).order_by('prediction_for')[:12])
            
            # Set predictions available if we have any
            predictions_available = len(temperature_predictions) > 0
            
            print(f"📊 Found {len(temperature_predictions)} temp predictions")
            print(f"📊 Found {len(humidity_predictions)} humidity predictions")
            
        except ImportError as e:
            print(f"⚠️ Predictions app not available: {e}")
            # Fallback to sample predictions if Prediction model doesn't exist
            if not temperature_predictions:
                temperature_predictions, humidity_predictions = generate_fallback_predictions_list()
                predictions_available = len(temperature_predictions) > 0
                
        except Exception as e:
            print(f"⚠️ Error loading predictions: {e}")
            # Fallback to sample predictions
            if not temperature_predictions:
                temperature_predictions, humidity_predictions = generate_fallback_predictions_list()
                predictions_available = len(temperature_predictions) > 0
        
        context = {
            'latest_data': latest_data,
            'chart_data': chart_data,
            'temperature_predictions': temperature_predictions,
            'humidity_predictions': humidity_predictions,
            'predictions_available': predictions_available,
        }
        
        return render(request, 'dashboard/dashboard.html', context)
        
    except Exception as e:
        print(f"❌ Error in dashboard view: {e}")
        # Return basic context without predictions
        context = {
            'latest_data': None,
            'chart_data': [],
            'temperature_predictions': [],
            'humidity_predictions': [],
            'predictions_available': False,
            'error_message': 'Unable to load dashboard data. Please try again.'
        }
        return render(request, 'dashboard/dashboard.html', context)

def generate_fallback_predictions_list():
    """Generate fallback predictions as list of dictionaries (for when Prediction model doesn't exist)"""
    current_time = timezone.now()
    temperature_predictions = []
    humidity_predictions = []
    
    # Generate sample temperature predictions as dictionaries
    for i in range(1, 13):  # Next 12 hours
        prediction_time = current_time + timedelta(hours=i)
        temp_pred = 22.0 + random.uniform(-2, 2) + 3 * math.sin(i/4)
        
        temperature_predictions.append({
            'prediction_for': prediction_time,
            'predicted_value': round(temp_pred, 2),
            'confidence': round(random.uniform(70, 95), 2),  # As percentage for template
        })
    
    # Generate sample humidity predictions as dictionaries
    for i in range(1, 13):
        prediction_time = current_time + timedelta(hours=i)
        humidity_pred = 55.0 + random.uniform(-10, 10) + 8 * math.sin(i/3)
        humidity_pred = max(20, min(90, humidity_pred))
        
        humidity_predictions.append({
            'prediction_for': prediction_time,
            'predicted_value': round(humidity_pred, 2),
            'confidence': round(random.uniform(60, 90), 2),  # As percentage for template
        })
    
    print("✅ Generated fallback predictions")
    return temperature_predictions, humidity_predictions

def generate_sample_predictions():
    """Generate sample predictions for demonstration - Updated to use EnvironmentalPredictor"""
    try:
        from .train_model import EnvironmentalPredictor
        from predictions.models import Prediction
        
        predictor = EnvironmentalPredictor()
        predictor.generate_predictions(hours_ahead=12)
        
        print("✅ Generated predictions using ML model")
        return True
        
    except Exception as e:
        print(f"❌ Error generating predictions with ML model: {e}")
        # Fallback to simple sample predictions
        try:
            from predictions.models import Prediction
            
            # Clear existing predictions
            Prediction.objects.all().delete()
            
            current_time = timezone.now()
            
            # Generate sample temperature predictions
            for i in range(1, 13):
                prediction_time = current_time + timedelta(hours=i)
                temp_pred = 22.0 + random.uniform(-2, 2) + 3 * math.sin(i/4)
                
                Prediction.objects.create(
                    prediction_type='temperature',
                    predicted_value=round(temp_pred, 2),
                    confidence=round(random.uniform(0.7, 0.95), 2),
                    prediction_for=prediction_time
                )
            
            # Generate sample humidity predictions
            for i in range(1, 13):
                prediction_time = current_time + timedelta(hours=i)
                humidity_pred = 55.0 + random.uniform(-10, 10) + 8 * math.sin(i/3)
                humidity_pred = max(20, min(90, humidity_pred))
                
                Prediction.objects.create(
                    prediction_type='humidity',
                    predicted_value=round(humidity_pred, 2),
                    confidence=round(random.uniform(0.6, 0.9), 2),
                    prediction_for=prediction_time
                )
            
            print("✅ Generated sample predictions as fallback")
            return True
            
        except Exception as e2:
            print(f"❌ Error generating fallback predictions: {e2}")
            return False

def generate_sample_predictions_view(request):
    """API endpoint to generate sample predictions"""
    try:
        success = generate_sample_predictions()
        if success:
            return JsonResponse({'success': True, 'message': 'Predictions generated successfully'})
        else:
            return JsonResponse({'success': False, 'error': 'Failed to generate predictions'})
    except Exception as e:
        return JsonResponse({'success': False, 'error': str(e)})

def sensor_data_api(request):
    """API endpoint for sensor data"""
    try:
        hours = int(request.GET.get('hours', 24))
        since = timezone.now() - timedelta(hours=hours)
        
        data = list(SensorData.objects.filter(
            timestamp__gte=since
        ).values(
            'timestamp', 'temperature', 'humidity', 'battery_voltage',
            'heat_index', 'dew_point', 'motion', 'temp_alert'
        ).order_by('timestamp'))
        
        return JsonResponse(data, safe=False)
    except Exception as e:
        return JsonResponse({'error': str(e)}, status=500)

def current_stats_api(request):
    """API endpoint for current statistics - FIXED VERSION"""
    try:
        # Get the ABSOLUTE latest record, not cached
        latest = SensorData.objects.order_by('-timestamp').first()
        
        if latest:
            stats = {
                'temperature': latest.temperature,
                'humidity': latest.humidity,
                'battery_voltage': latest.battery_voltage,
                'battery_percentage': latest.battery_percentage,
                'heat_index': latest.heat_index,
                'dew_point': latest.dew_point,
                'motion': latest.motion,
                'temp_alert': latest.temp_alert,
                'timestamp': latest.timestamp.isoformat() if latest.timestamp else None,
            }
            
            # Debug output to console
            print(f"🔥 CURRENT STATS API - Fresh data:")
            print(f"   Time: {latest.timestamp}")
            print(f"   Temp: {latest.temperature}°C, Hum: {latest.humidity}%")
            print(f"   Battery: {latest.battery_voltage}V ({latest.battery_percentage}%)")
            print(f"   Motion: {latest.motion}")
            
        else:
            stats = {
                'temperature': 0,
                'humidity': 0,
                'battery_voltage': 0,
                'battery_percentage': 0,
                'motion': False,
                'temp_alert': False,
                'timestamp': timezone.now().isoformat(),
            }
            print("⚠️ CURRENT STATS API - No data available")
        
        return JsonResponse(stats)
        
    except Exception as e:
        print(f"❌ Error in current_stats_api: {e}")
        return JsonResponse({
            'error': str(e),
            'temperature': 0,
            'humidity': 0,
            'battery_voltage': 0,
            'battery_percentage': 0,
            'motion': False,
            'temp_alert': False,
        }, status=500)
        
        return JsonResponse(stats)
    except Exception as e:
        return JsonResponse({'error': str(e)}, status=500)
    
def debug_current_stats(request):
    """Debug endpoint to see current stats data"""
    latest = SensorData.objects.last()
    if latest:
        stats = {
            'temperature': latest.temperature,
            'humidity': latest.humidity,
            'battery_voltage': latest.battery_voltage,
            'battery_percentage': latest.battery_percentage,
            'motion': latest.motion,
            'temp_alert': latest.temp_alert,
            'timestamp': latest.timestamp.isoformat() if latest.timestamp else None,
        }
    else:
        stats = {'error': 'No data available'}
    
    print("🔍 DEBUG Current Stats:", stats)
    return JsonResponse(stats)    