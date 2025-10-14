import paho.mqtt.client as mqtt
import json
import requests
from django.utils import timezone
from .models import SensorData
from django.conf import settings
import math
import threading

class MQTTService:
    def __init__(self):
        self.broker = "eu1.cloud.thethings.network"
        self.port = 1883
        self.username = "bd-test-app2@ttn"
        self.password = "NNSXS.NGFSXX4UXDX55XRIDQZS6LPR4OJXKIIGSZS56CQ.6O4WUAUHFUAHSTEYRWJX6DDO7TL2IBLC7EV2LS4EHWZOOEPCEUOA"
        self.device_id = "lht65n-01-temp-humidity-sensor"
        self.client = None
        self.is_connected = False

    def calculate_heat_index(self, temp, humidity):
        """Calculate heat index"""
        try:
            if temp is None or humidity is None:
                return None
            T = temp
            R = humidity
            hi = T + 0.33 * (R / 100) * (T - 14.4) + 32.0
            return round(hi, 2)
        except:
            return None

    def calculate_dew_point(self, temp, humidity):
        """Calculate dew point temperature"""
        try:
            if temp is None or humidity is None:
                return None
            a = 17.27
            b = 237.7
            alpha = ((a * temp) / (b + temp)) + math.log(humidity/100.0)
            dew_point = (b * alpha) / (a - alpha)
            return round(dew_point, 2)
        except:
            return None

    def approximate_battery_percentage(self, voltage):
        """Approximate battery percentage"""
        try:
            if voltage is None:
                return None
            full_voltage = 3.7
            empty_voltage = 3.0
            percentage = max(0, min(100, (voltage - empty_voltage) / (full_voltage - empty_voltage) * 100))
            return round(percentage, 1)
        except:
            return None

    def on_connect(self, client, userdata, flags, rc):
        if rc == 0:
            print("✅ Connected to TTN MQTT broker!")
            self.is_connected = True
            topic = f"v3/{self.username}/devices/{self.device_id}/up"
            client.subscribe(topic)
            print(f"Subscribed to topic: {topic}")
        else:
            print(f"❌ Failed to connect to MQTT broker with code: {rc}")

    def on_message(self, client, userdata, msg):
        try:
            payload = json.loads(msg.payload.decode())
            self.process_sensor_data(payload)
        except Exception as e:
            print(f"Error processing MQTT message: {e}")

    def process_sensor_data(self, payload):
        """Process sensor data and save to database"""
        try:
            uplink = payload['uplink_message']
            decoded = uplink['decoded_payload']
            
            temperature = decoded.get('field5')
            humidity = decoded.get('field3')
            battery_voltage = decoded.get('field1')
            motion_status = decoded.get('Exti_pin_level', 'No activity')
            
            motion_numeric = 1 if motion_status == "Activity" else 0
            heat_index = self.calculate_heat_index(temperature, humidity)
            dew_point = self.calculate_dew_point(temperature, humidity)
            battery_percentage = self.approximate_battery_percentage(battery_voltage)
            temp_alert = temperature > 25 if temperature else False
            
            # Save to database
            sensor_data = SensorData(
                temperature=temperature,
                humidity=humidity,
                motion=motion_numeric,
                battery_voltage=battery_voltage,
                heat_index=heat_index,
                dew_point=dew_point,
                battery_percentage=battery_percentage,
                temp_alert=temp_alert
            )
            sensor_data.save()
            
            print(f"✅ Saved sensor data: {temperature}°C, {humidity}%")
            
        except Exception as e:
            print(f"Error saving sensor data: {e}")

    def start(self):
        """Start MQTT client"""
        self.client = mqtt.Client()
        self.client.username_pw_set(self.username, self.password)
        self.client.on_connect = self.on_connect
        self.client.on_message = self.on_message
        
        try:
            self.client.connect(self.broker, self.port, 60)
            self.client.loop_start()
        except Exception as e:
            print(f"Error starting MQTT client: {e}")

    def stop(self):
        """Stop MQTT client"""
        if self.client:
            self.client.loop_stop()
            self.client.disconnect()

# Global MQTT service instance
mqtt_service = MQTTService()