import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error
import joblib
import os
from django.utils import timezone
from datetime import timedelta, datetime
from dashboard.models import SensorData
import math

class EnvironmentalPredictor:
    def __init__(self):
        self.temperature_model = None
        self.humidity_model = None
        self.models_dir = 'media/models/'
        self.dataset_path = 'cleaned_feeds.csv'  # Path to your dataset
        os.makedirs(self.models_dir, exist_ok=True)

    def load_dataset(self):
        """Load and prepare the historical dataset"""
        try:
            df = pd.read_csv(self.dataset_path)
            df['timestamp'] = pd.to_datetime(df['timestamp'])
            df = df.sort_values('timestamp')
            
            # Rename columns to match your model expectations
            df = df.rename(columns={'Humidity(%)': 'humidity', 'Temperature': 'temperature'})
            
            print(f"✅ Loaded dataset with {len(df)} records")
            return df
            
        except Exception as e:
            print(f"❌ Error loading dataset: {e}")
            return None

    def prepare_features(self, df):
        """Prepare features for training"""
        df = df.copy()
        df['timestamp'] = pd.to_datetime(df['timestamp'])
        df['hour'] = df['timestamp'].dt.hour
        df['day_of_week'] = df['timestamp'].dt.dayofweek
        df['month'] = df['timestamp'].dt.month
        df['day_of_year'] = df['timestamp'].dt.dayofyear
        
        # Create lag features
        for lag in [1, 2, 3]:
            df[f'temp_lag_{lag}'] = df['temperature'].shift(lag)
            df[f'humidity_lag_{lag}'] = df['humidity'].shift(lag)
        
        # Rolling statistics
        df['temp_rolling_mean_3'] = df['temperature'].rolling(3, min_periods=1).mean()
        df['humidity_rolling_mean_3'] = df['humidity'].rolling(3, min_periods=1).mean()
        
        # Time-based features
        df['sin_hour'] = np.sin(2 * np.pi * df['hour']/24)
        df['cos_hour'] = np.cos(2 * np.pi * df['hour']/24)
        
        return df.dropna()

    def train_models(self):
        """Train temperature and humidity prediction models using the dataset"""
        try:
            # Load the dataset
            df = self.load_dataset()
            if df is None or len(df) == 0:
                print("❌ No dataset available for training")
                return False
            
            # Prepare features
            df_processed = self.prepare_features(df)
            
            if len(df_processed) < 20:
                print("❌ Not enough processed data for training")
                return False
            
            # Features for training
            feature_columns = ['hour', 'day_of_week', 'month', 'day_of_year', 
                             'temp_lag_1', 'temp_lag_2', 'temp_lag_3',
                             'humidity_lag_1', 'humidity_lag_2', 'humidity_lag_3',
                             'temp_rolling_mean_3', 'humidity_rolling_mean_3',
                             'sin_hour', 'cos_hour']
            
            X = df_processed[feature_columns]
            
            # Train temperature model
            y_temp = df_processed['temperature']
            X_train, X_test, y_temp_train, y_temp_test = train_test_split(
                X, y_temp, test_size=0.2, random_state=42
            )
            
            self.temperature_model = RandomForestRegressor(
                n_estimators=100, 
                random_state=42,
                max_depth=10,
                min_samples_split=5
            )
            self.temperature_model.fit(X_train, y_temp_train)
            
            # Evaluate temperature model
            temp_pred = self.temperature_model.predict(X_test)
            temp_mae = mean_absolute_error(y_temp_test, temp_pred)
            print(f"✅ Temperature Model MAE: {temp_mae:.2f}°C")
            
            # Train humidity model
            y_humidity = df_processed['humidity']
            X_train_h, X_test_h, y_humidity_train, y_humidity_test = train_test_split(
                X, y_humidity, test_size=0.2, random_state=42
            )
            
            self.humidity_model = RandomForestRegressor(
                n_estimators=100, 
                random_state=42,
                max_depth=10,
                min_samples_split=5
            )
            self.humidity_model.fit(X_train_h, y_humidity_train)
            
            # Evaluate humidity model
            humidity_pred = self.humidity_model.predict(X_test_h)
            humidity_mae = mean_absolute_error(y_humidity_test, humidity_pred)
            print(f"✅ Humidity Model MAE: {humidity_mae:.2f}%")
            
            # Save models
            joblib.dump(self.temperature_model, f'{self.models_dir}temperature_model.joblib')
            joblib.dump(self.humidity_model, f'{self.models_dir}humidity_model.joblib')
            
            print("✅ Models trained and saved successfully using historical dataset")
            return True
            
        except Exception as e:
            print(f"❌ Error training models: {e}")
            return False

    def create_future_features(self, last_timestamp, last_temperature, last_humidity, hours_ahead):
        """Create feature set for future predictions"""
        future_features = []
        
        for i in range(1, hours_ahead + 1):
            prediction_time = last_timestamp + timedelta(hours=i)
            
            features = {
                'hour': prediction_time.hour,
                'day_of_week': prediction_time.weekday(),
                'month': prediction_time.month,
                'day_of_year': prediction_time.timetuple().tm_yday,
                'sin_hour': np.sin(2 * np.pi * prediction_time.hour/24),
                'cos_hour': np.cos(2 * np.pi * prediction_time.hour/24),
                # For lag features, we'll use the last known values (simplified approach)
                'temp_lag_1': last_temperature,
                'temp_lag_2': last_temperature,
                'temp_lag_3': last_temperature,
                'humidity_lag_1': last_humidity,
                'humidity_lag_2': last_humidity,
                'humidity_lag_3': last_humidity,
                'temp_rolling_mean_3': last_temperature,
                'humidity_rolling_mean_3': last_humidity,
            }
            
            future_features.append(features)
        
        return pd.DataFrame(future_features)

    def generate_predictions(self, hours_ahead=12):
        """Generate predictions for future hours using trained models"""
        from predictions.models import Prediction
        
        try:
            # Clear old predictions
            Prediction.objects.all().delete()
            
            # Try to load trained models first
            if not self.load_trained_models():
                # If no trained models, train them first
                print("🔄 No trained models found. Training new models...")
                if not self.train_models():
                    print("❌ Could not train models, using fallback")
                    self.generate_fallback_predictions(hours_ahead)
                    return
            
            # Get the most recent data for prediction context
            recent_data = SensorData.objects.filter(
                temperature__isnull=False,
                humidity__isnull=False
            ).order_by('-timestamp').first()
            
            if recent_data:
                last_temperature = recent_data.temperature
                last_humidity = recent_data.humidity
                last_timestamp = recent_data.timestamp
            else:
                # Use dataset statistics if no recent data
                df = self.load_dataset()
                last_temperature = df['temperature'].mean()
                last_humidity = df['humidity'].mean()
                last_timestamp = timezone.now()
                print("⚠️ Using dataset averages for prediction context")
            
            # Create features for future predictions
            future_df = self.create_future_features(
                last_timestamp, last_temperature, last_humidity, hours_ahead
            )
            
            # Ensure feature order matches training
            feature_columns = ['hour', 'day_of_week', 'month', 'day_of_year', 
                             'temp_lag_1', 'temp_lag_2', 'temp_lag_3',
                             'humidity_lag_1', 'humidity_lag_2', 'humidity_lag_3',
                             'temp_rolling_mean_3', 'humidity_rolling_mean_3',
                             'sin_hour', 'cos_hour']
            
            future_df = future_df[feature_columns]
            
            # Generate predictions
            temp_predictions = self.temperature_model.predict(future_df)
            humidity_predictions = self.humidity_model.predict(future_df)
            
            # Create prediction records
            for i, (temp_pred, humidity_pred) in enumerate(zip(temp_predictions, humidity_predictions)):
                prediction_time = last_timestamp + timedelta(hours=i+1)
                
                # Calculate confidence based on time of day (higher confidence for typical hours)
                hour = prediction_time.hour
                confidence_base = 0.7 + (0.2 if 6 <= hour <= 22 else 0.1)  # Higher confidence during daytime
                
                Prediction.objects.create(
                    prediction_type='temperature',
                    predicted_value=round(float(temp_pred), 2),
                    confidence=round(confidence_base + np.random.uniform(-0.1, 0.1), 2),
                    prediction_for=prediction_time
                )
                
                Prediction.objects.create(
                    prediction_type='humidity',
                    predicted_value=round(float(humidity_pred), 2),
                    confidence=round(confidence_base - 0.1 + np.random.uniform(-0.1, 0.1), 2),
                    prediction_for=prediction_time
                )
            
            print(f"✅ Generated {hours_ahead * 2} predictions using trained models")
            
        except Exception as e:
            print(f"❌ Error generating predictions with models: {e}")
            self.generate_fallback_predictions(hours_ahead)

    def load_trained_models(self):
        """Load pre-trained models if they exist"""
        try:
            temp_model_path = f'{self.models_dir}temperature_model.joblib'
            humidity_model_path = f'{self.models_dir}humidity_model.joblib'
            
            if os.path.exists(temp_model_path) and os.path.exists(humidity_model_path):
                self.temperature_model = joblib.load(temp_model_path)
                self.humidity_model = joblib.load(humidity_model_path)
                print("✅ Loaded pre-trained models")
                return True
            return False
        except Exception as e:
            print(f"❌ Error loading trained models: {e}")
            return False

    def generate_fallback_predictions(self, hours_ahead=12):
        """Generate basic predictions when models fail"""
        from predictions.models import Prediction
        
        current_time = timezone.now()
        df = self.load_dataset()
        
        if df is not None:
            # Use dataset patterns for fallback
            avg_temp = df['temperature'].mean()
            avg_humidity = df['humidity'].mean()
            temp_std = df['temperature'].std()
            humidity_std = df['humidity'].std()
        else:
            # Default values if no dataset
            avg_temp = 25.0
            avg_humidity = 65.0
            temp_std = 2.0
            humidity_std = 10.0
        
        for i in range(1, hours_ahead + 1):
            prediction_time = current_time + timedelta(hours=i)
            
            # Time-based patterns
            hour = prediction_time.hour
            temp_variation = 3 * math.sin((hour - 14) * math.pi / 12)  # Peak at 2 PM
            humidity_variation = -15 * math.sin((hour - 6) * math.pi / 12)  # Lower during day
            
            temp_pred = avg_temp + temp_variation + np.random.normal(0, temp_std * 0.3)
            humidity_pred = max(20, min(90, avg_humidity + humidity_variation + np.random.normal(0, humidity_std * 0.3)))
            
            Prediction.objects.create(
                prediction_type='temperature',
                predicted_value=round(temp_pred, 2),
                confidence=round(0.6 + np.random.uniform(0, 0.2), 2),
                prediction_for=prediction_time
            )
            
            Prediction.objects.create(
                prediction_type='humidity',
                predicted_value=round(humidity_pred, 2),
                confidence=round(0.5 + np.random.uniform(0, 0.25), 2),
                prediction_for=prediction_time
            )
        
        print(f"✅ Generated {hours_ahead * 2} fallback predictions")