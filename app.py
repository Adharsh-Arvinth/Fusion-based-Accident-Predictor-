import streamlit as st
import cv2
import pandas as pd
import numpy as np
import time
import random
from datetime import datetime
import threading
import json
from collections import deque
import warnings

# Suppress FutureWarning from deprecated google.generativeai
warnings.filterwarnings('ignore', category=FutureWarning, module='google.generativeai')

# Try to import Gemini AI (try new package first, then fallback to old)
try:
    from google import genai
    gemini_available = True
    using_new_api = True
except ImportError:
    try:
        import google.generativeai as genai
        gemini_available = True
        using_new_api = False
    except ImportError:
        gemini_available = False
        using_new_api = False

# Try to import Voice Library
try:
    import pyttsx3
    voice_enabled = True
except ImportError:
    voice_enabled = False

# Try to import audio processing
try:
    import sounddevice as sd
    import librosa
    audio_enabled = True
except ImportError:
    audio_enabled = False

# ==========================================
# 1. VOICE ALERT ENGINE (Runs in Background)
# ==========================================
def speak_alert(text):
    if not voice_enabled: return
    def run_speech():
        try:
            # Re-initialize in thread to prevent crashes
            engine = pyttsx3.init()
            engine.setProperty('rate', 160) # Speed of speech
            engine.say(text)
            engine.runAndWait()
        except Exception as e:
            pass
    threading.Thread(target=run_speech, daemon=True).start()

# ==========================================
# 2. AI SETUP (Official Google Library)
# ==========================================
# Try to get API key from Streamlit secrets (for cloud), fallback to hardcoded (for local)
try:
    GEMINI_API_KEY = st.secrets["GEMINI_API_KEY"]
except Exception:
    GEMINI_API_KEY = "AIzaSyADPgT3Qk-OBcQwqIpgc9oQ8XARsy4Rpt8"

def get_gemini_response(prompt):
    """Get AI response from Gemini API"""
    if not gemini_available:
        return "⚠️ Gemini AI not available. Install: pip install google-generativeai"
    
    # List of models to try in order (with full paths)
    models_to_try = [
        'models/gemini-2.5-flash',
        'models/gemini-flash-latest',
        'models/gemini-pro-latest',
        'models/gemini-2.0-flash',
        'gemini-2.5-flash',
        'gemini-flash-latest',
        'gemini-pro-latest'
    ]
    
    # Try old google.generativeai package (it's what we have installed)
    try:
        import google.generativeai as old_genai
        old_genai.configure(api_key=GEMINI_API_KEY)
        
        for model_name in models_to_try:
            try:
                model = old_genai.GenerativeModel(model_name)
                response = model.generate_content(prompt)
                return response.text
            except Exception as e:
                continue  # Try next model
                
    except Exception as e:
        pass
    
    # If all attempts fail, return helpful error
    return """⚠️ AI Analysis Unavailable

Unable to connect to Gemini AI. This could be due to:
• API key issues
• Network connectivity
• Model availability

Your trip data has been saved and you can still:
✅ View trip statistics below
✅ Download CSV data
✅ Review telemetry logs

Trip completed successfully - AI coaching will be available once connection is restored."""

# ==========================================
# 3. OPENCV SETUP
# ==========================================
face_cascade = None
eye_cascade = None
try:
    haarcascade_path = cv2.data.haarcascades if hasattr(cv2, 'data') and cv2.data.haarcascades else ''
    face_cascade = cv2.CascadeClassifier(haarcascade_path + 'haarcascade_frontalface_default.xml')
    eye_cascade = cv2.CascadeClassifier(haarcascade_path + 'haarcascade_eye.xml')
    if face_cascade.empty():
        face_cascade = None
    if eye_cascade.empty():
        eye_cascade = None
except Exception:
    face_cascade = None
    eye_cascade = None

# ==========================================
# 4. ENHANCED FUSION LOGIC WITH ALL INPUTS
# ==========================================
def calculate_fusion_risk(driver_state, vehicle_dynamics, audio_stress, gps_context, history):
    """
    Multi-modal fusion combining:
    - Driver facial/eye state + emotion/stress
    - Vehicle dynamics (steering, braking, acceleration)
    - Audio stress levels + shouting detection
    - GPS road risk context
    - Historical driving patterns
    """
    risk_score = 0
    risk_factors = []
    
    # Driver State (40% weight)
    if driver_state["eye_state"] == "DROWSY":
        risk_score += 30
        risk_factors.append("Driver drowsiness detected")
    
    # Facial Stress Analysis (NEW)
    facial_stress = driver_state.get("facial_stress", {})
    stress_state = facial_stress.get("stress_state", "CALM")
    
    if stress_state == "RAGE":
        risk_score += 20
        risk_factors.append("⚠️ DRIVER IN RAGE STATE - Extreme anger detected")
    elif stress_state == "STRESSED":
        risk_score += 10
        risk_factors.append("Driver showing stress/frustration")
    
    if facial_stress.get("jaw_clench"):
        risk_score += 5
        risk_factors.append("Jaw clenching detected")
    
    if facial_stress.get("stress_duration", 0) > 10:
        risk_score += 5
        risk_factors.append("Prolonged stress state")
    
    # Legacy facial tension (keep for compatibility)
    if driver_state.get("facial_tension", 0) > 0.7:
        risk_score += 5
        risk_factors.append("High facial tension")
    
    # Vehicle Dynamics (30% weight)
    if vehicle_dynamics["harsh_braking"]:
        risk_score += 15
        risk_factors.append("Harsh braking detected")
    if vehicle_dynamics["aggressive_steering"]:
        risk_score += 10
        risk_factors.append("Aggressive steering")
    if vehicle_dynamics["speed_fluctuation"] > 20:
        risk_score += 5
        risk_factors.append("Erratic speed changes")
    
    # Audio Stress (15% weight)
    if audio_stress["stress_level"] == "HIGH":
        risk_score += 10
        risk_factors.append("Voice stress detected")
    if audio_stress["shouting_detected"]:
        risk_score += 8
        risk_factors.append("🔊 DRIVER SHOUTING DETECTED")
    if audio_stress.get("shout_count", 0) > 3:
        risk_score += 5
        risk_factors.append(f"Multiple shouting incidents ({audio_stress['shout_count']})")
    
    # GPS Context (15% weight)
    if gps_context["zone_risk"] == "High":
        risk_score += 10
        risk_factors.append("High-risk accident zone")
    if gps_context["speed"] > gps_context["speed_limit"]:
        risk_score += 5
        risk_factors.append(f"Speeding ({gps_context['speed']} km/h in {gps_context['speed_limit']} zone)")
    if gps_context["weather_risk"] == "Poor":
        risk_score += 5
        risk_factors.append("Poor weather conditions")
    
    # Historical Pattern Adjustment
    if history["aggressive_driver"]:
        risk_score += 5
        risk_factors.append("Aggressive driving history")
    
    # Determine risk level
    if risk_score > 75:
        return "CRITICAL", min(risk_score, 100), "bad", risk_factors
    elif risk_score > 50:
        return "HIGH", risk_score, "warn", risk_factors
    elif risk_score > 25:
        return "MEDIUM", risk_score, "warn", risk_factors
    else:
        return "LOW", risk_score, "good", risk_factors

def get_user_history():
    """Load or simulate historical driving patterns"""
    return {
        "aggressive_driver": True,
        "past_harsh_braking": 15,
        "near_miss_count": 3,
        "avg_stress_score": 6.5
    }

# ==========================================
# 5. VEHICLE DYNAMICS SIMULATOR
# ==========================================
class VehicleDynamicsSimulator:
    def __init__(self):
        self.steering_history = deque(maxlen=10)
        self.brake_history = deque(maxlen=10)
        self.speed_history = deque(maxlen=10)
        
    def update(self, speed, risk_level):
        """Simulate vehicle sensor data"""
        # Simulate steering jerks
        steering_angle = random.uniform(-5, 5)
        if risk_level > 50:
            steering_angle += random.uniform(-15, 15)
        self.steering_history.append(abs(steering_angle))
        
        # Simulate braking
        brake_force = 0
        if random.random() < 0.1 or risk_level > 60:
            brake_force = random.uniform(0.5, 1.0)
        self.brake_history.append(brake_force)
        
        # Track speed
        self.speed_history.append(speed)
        
        # Calculate metrics
        aggressive_steering = max(self.steering_history) > 10 if self.steering_history else False
        harsh_braking = max(self.brake_history) > 0.7 if self.brake_history else False
        speed_fluctuation = max(self.speed_history) - min(self.speed_history) if len(self.speed_history) > 1 else 0
        
        return {
            "aggressive_steering": aggressive_steering,
            "harsh_braking": harsh_braking,
            "speed_fluctuation": speed_fluctuation,
            "steering_angle": steering_angle,
            "brake_force": brake_force
        }

# ==========================================
# 6. AUDIO STRESS ANALYZER
# ==========================================
class AudioStressAnalyzer:
    def __init__(self):
        self.stress_history = deque(maxlen=20)
        self.shout_count = 0
        
    def analyze(self, risk_level):
        """Simulate audio stress detection - MORE SENSITIVE"""
        # In real implementation, use MFCC features from microphone
        base_stress = random.uniform(0.4, 0.7)  # Increased from 0.2-0.5
        if risk_level > 40:  # Lowered threshold from 50
            base_stress += random.uniform(0.4, 0.6)  # Increased
        
        stress_level = "LOW"
        if base_stress > 0.6:  # Lowered from 0.7
            stress_level = "HIGH"
        elif base_stress > 0.4:  # Lowered from 0.5
            stress_level = "MEDIUM"
        
        # MORE FREQUENT shouting detection
        shouting = risk_level > 40 and random.random() < 0.5  # Increased from 0.3, lowered threshold
        if shouting:
            self.shout_count += 1
        
        # Detect voice patterns - MORE AGGRESSIVE
        voice_pitch = random.uniform(150, 350) if shouting else random.uniform(80, 200)  # Higher pitch when shouting
        voice_volume = random.uniform(75, 95) if shouting else random.uniform(40, 70)  # Louder when shouting
        
        self.stress_history.append(base_stress)
        
        return {
            "stress_level": stress_level,
            "stress_value": base_stress,
            "shouting_detected": shouting,
            "shout_count": self.shout_count,
            "cabin_noise": voice_volume,
            "voice_pitch": voice_pitch,
            "voice_agitation": "HIGH" if voice_pitch > 180 else "NORMAL"  # Lowered from 200
        }

# ==========================================
# 6B. FACIAL EMOTION & STRESS DETECTOR
# ==========================================
class FacialStressDetector:
    def __init__(self):
        self.stress_history = deque(maxlen=30)
        self.emotion_history = deque(maxlen=20)
        
    def analyze_facial_stress(self, face_detected, eye_state, risk_level):
        """
        Analyze facial expressions for stress levels:
        - CALM: Relaxed, normal driving
        - STRESSED: Tension, frowning, tight jaw
        - RAGE: Extreme anger, shouting face, aggressive
        """
        if not face_detected:
            return {
                "stress_state": "UNKNOWN",
                "stress_level": 0.0,
                "emotion": "Not Detected",
                "facial_tension": 0.0,
                "jaw_clench": False,
                "eyebrow_furrow": False,
                "mouth_tension": False
            }
        
        # MORE SENSITIVE stress detection - triggers more easily
        base_stress = random.uniform(0.3, 0.6)  # Increased from 0.1-0.3
        
        # Increase stress based on driving conditions (MORE AGGRESSIVE)
        if eye_state == "DROWSY":
            base_stress += random.uniform(0.3, 0.5)  # Increased
        if risk_level > 40:  # Lowered threshold from 50
            base_stress += random.uniform(0.4, 0.6)  # Increased
        
        # More frequent facial micro-expressions
        jaw_clench = base_stress > 0.5 and random.random() < 0.6  # Increased from 0.4
        eyebrow_furrow = base_stress > 0.4 and random.random() < 0.7  # Increased from 0.5
        mouth_tension = base_stress > 0.6 and random.random() < 0.7  # Increased from 0.6
        
        # MORE SENSITIVE stress state thresholds
        if base_stress > 0.7 or (jaw_clench and eyebrow_furrow and mouth_tension):
            stress_state = "RAGE"
            emotion = "Angry/Furious"
        elif base_stress > 0.4 or (jaw_clench or eyebrow_furrow):  # Lowered from 0.5
            stress_state = "STRESSED"
            emotion = "Tense/Frustrated"
        else:
            stress_state = "CALM"
            emotion = "Relaxed/Neutral"
        
        self.stress_history.append(base_stress)
        self.emotion_history.append(stress_state)
        
        # Calculate average stress over time
        avg_stress = sum(self.stress_history) / len(self.stress_history) if self.stress_history else 0
        
        return {
            "stress_state": stress_state,
            "stress_level": base_stress,
            "avg_stress": avg_stress,
            "emotion": emotion,
            "facial_tension": base_stress,
            "jaw_clench": jaw_clench,
            "eyebrow_furrow": eyebrow_furrow,
            "mouth_tension": mouth_tension,
            "stress_duration": len([s for s in self.emotion_history if s in ["STRESSED", "RAGE"]])
        }

# ==========================================
# 7. GPS & ACCIDENT ZONE DATABASE
# ==========================================
ACCIDENT_PRONE_ZONES = [
    {"lat": 12.9165, "lon": 79.1325, "name": "Highway Junction", "risk": "High", "speed_limit": 60},
    {"lat": 12.9180, "lon": 79.1340, "name": "Sharp Curve", "risk": "High", "speed_limit": 40},
    {"lat": 12.9150, "lon": 79.1310, "name": "School Zone", "risk": "Medium", "speed_limit": 30},
    {"lat": 12.9200, "lon": 79.1360, "name": "City Center", "risk": "Low", "speed_limit": 50},
]

def get_gps_context(lat, lon, speed):
    """Determine road risk based on GPS location"""
    # Find nearest zone
    min_dist = float('inf')
    nearest_zone = ACCIDENT_PRONE_ZONES[-1]
    
    for zone in ACCIDENT_PRONE_ZONES:
        dist = ((lat - zone["lat"])**2 + (lon - zone["lon"])**2)**0.5
        if dist < min_dist:
            min_dist = dist
            nearest_zone = zone
    
    # Simulate weather
    weather_conditions = ["Clear", "Clear", "Clear", "Poor"]
    weather = random.choice(weather_conditions)
    
    return {
        "zone_name": nearest_zone["name"],
        "zone_risk": nearest_zone["risk"],
        "speed_limit": nearest_zone["speed_limit"],
        "speed": speed,
        "weather_risk": weather,
        "lat": lat,
        "lon": lon
    }

# ==========================================
# 8. EMERGENCY CALLING SYSTEM
# ==========================================
def trigger_emergency_call(location, risk_factors, emergency_contacts):
    """Simulate emergency services and contacts calling"""
    emergency_data = {
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "location": f"Lat: {location['lat']:.4f}, Lon: {location['lon']:.4f}",
        "zone": location.get("zone_name", "Unknown"),
        "risk_factors": risk_factors,
        "contacts_notified": [f"{c['name']} ({c['phone']})" for c in emergency_contacts],
        "status": "EMERGENCY SERVICES & CONTACTS NOTIFIED"
    }
    
    # In real implementation: Make actual emergency calls via cellular/API
    # For now, log and display
    return emergency_data

# ==========================================
# 9. UI & DASHBOARD CONFIG
# ==========================================
st.set_page_config(page_title="Fusion-Based Accident Predictor", layout="wide", initial_sidebar_state="expanded")

st.markdown("""
<style>
/* Global Styles - Premium Professional Theme with High Contrast */
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&display=swap');

.stApp { 
    background: #f5f7fa;
    font-family: 'Inter', -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif;
}

/* Force all text to be dark and readable */
p, span, div, label {
    color: #1e293b !important;
}

/* Header Styles - Premium Design */
.main-header {
    background: linear-gradient(135deg, #1e40af 0%, #3b82f6 100%);
    padding: 40px 45px;
    border-radius: 16px;
    margin-bottom: 30px;
    box-shadow: 0 8px 32px rgba(30, 64, 175, 0.2);
    border: 1px solid rgba(255, 255, 255, 0.1);
}

.main-title { 
    font-size: 36px;
    color: #ffffff !important;
    font-weight: 800;
    margin-bottom: 10px;
    letter-spacing: -0.5px;
    text-shadow: 0 2px 4px rgba(0, 0, 0, 0.1);
}

.sub-title { 
    font-size: 15px;
    color: #e0e7ff !important;
    font-weight: 500;
    letter-spacing: 0.3px;
}

/* KPI Card Styles - Premium Cards with Better Contrast */
.kpi-card { 
    background: #ffffff;
    border-radius: 12px;
    padding: 24px;
    border: 1px solid #e2e8f0;
    margin-bottom: 16px;
    transition: all 0.3s cubic-bezier(0.4, 0, 0.2, 1);
    box-shadow: 0 2px 8px rgba(0, 0, 0, 0.06);
    position: relative;
    overflow: hidden;
}

.kpi-card::before {
    content: '';
    position: absolute;
    top: 0;
    left: 0;
    right: 0;
    height: 4px;
    background: linear-gradient(90deg, #3b82f6 0%, #2563eb 100%);
    opacity: 0;
    transition: opacity 0.3s ease;
}

.kpi-card:hover {
    box-shadow: 0 8px 24px rgba(0, 0, 0, 0.12);
    border-color: #cbd5e0;
    transform: translateY(-2px);
}

.kpi-card:hover::before {
    opacity: 1;
}

.kpi-icon {
    font-size: 32px;
    margin-bottom: 12px;
    display: block;
}

.kpi-value { 
    font-size: 32px;
    font-weight: 800;
    color: #0f172a !important;
    margin: 12px 0 8px 0;
    line-height: 1;
    letter-spacing: -0.5px;
}

.kpi-label { 
    font-size: 12px;
    color: #1e293b !important;
    text-transform: uppercase;
    letter-spacing: 1px;
    font-weight: 700;
    margin-bottom: 10px;
    display: block;
}

.kpi-sublabel {
    font-size: 13px;
    color: #334155 !important;
    margin-top: 10px;
    font-weight: 600;
}

.kpi-trend {
    font-size: 11px;
    font-weight: 600;
    margin-top: 5px;
}

/* Status Colors - Professional Palette */
.status-safe { 
    color: #059669;
}

.status-warning { 
    color: #d97706;
}

.status-danger { 
    color: #dc2626;
}

.badge-safe {
    background: #d1fae5;
    color: #065f46;
    padding: 4px 10px;
    border-radius: 12px;
    font-size: 11px;
    font-weight: 600;
    display: inline-block;
}

.badge-warning {
    background: #fef3c7;
    color: #92400e;
    padding: 4px 10px;
    border-radius: 12px;
    font-size: 11px;
    font-weight: 600;
    display: inline-block;
}

.badge-danger {
    background: #fee2e2;
    color: #991b1b;
    padding: 4px 10px;
    border-radius: 12px;
    font-size: 11px;
    font-weight: 600;
    display: inline-block;
}

/* Emergency Banner - Professional Alert */
.emergency-box { 
    background: linear-gradient(135deg, #fee2e2 0%, #fecaca 100%);
    color: #7f1d1d !important;
    padding: 24px 30px;
    border-radius: 12px;
    border-left: 5px solid #dc2626;
    margin-bottom: 24px;
    font-size: 15px;
    line-height: 1.7;
    box-shadow: 0 4px 16px rgba(220, 38, 38, 0.15);
    font-weight: 600;
}

.emergency-title {
    font-size: 18px;
    font-weight: 800;
    margin-bottom: 16px;
    color: #7f1d1d !important;
    letter-spacing: -0.3px;
}

.emergency-detail {
    margin: 8px 0;
    padding-left: 20px;
    position: relative;
}

.emergency-detail:before {
    content: "•";
    position: absolute;
    left: 5px;
    font-weight: bold;
}

/* Warning Box */
.warning-box {
    background: linear-gradient(135deg, #fef3c7 0%, #fde68a 100%);
    color: #78350f !important;
    padding: 20px 30px;
    border-radius: 12px;
    border-left: 5px solid #f59e0b;
    margin-bottom: 24px;
    font-size: 15px;
    box-shadow: 0 4px 16px rgba(245, 158, 11, 0.15);
    font-weight: 700;
}

/* Section Headers - Clean Design */
.section-header {
    background: linear-gradient(135deg, #ffffff 0%, #f8fafc 100%);
    padding: 16px 24px;
    border-radius: 12px;
    color: #0f172a !important;
    font-weight: 700;
    font-size: 15px;
    margin: 30px 0 20px 0;
    border: 1px solid #e2e8f0;
    border-left: 4px solid #3b82f6;
    box-shadow: 0 2px 8px rgba(0, 0, 0, 0.04);
    letter-spacing: -0.2px;
}

.section-icon {
    margin-right: 8px;
    opacity: 0.7;
}

/* Risk Factors Box */
.risk-factors-box {
    background: #ffffff;
    padding: 20px;
    border-radius: 8px;
    border: 1px solid #e2e8f0;
    box-shadow: 0 1px 3px rgba(0,0,0,0.04);
}

.risk-factor-header {
    color: #d97706;
    font-size: 13px;
    font-weight: 700;
    margin-bottom: 15px;
    text-transform: uppercase;
    letter-spacing: 0.5px;
}

.risk-factor-item {
    color: #0f172a !important;
    font-size: 14px;
    margin: 12px 0;
    padding: 14px 18px;
    background: linear-gradient(135deg, #fffbeb 0%, #fef3c7 100%);
    border-radius: 8px;
    border-left: 4px solid #f59e0b;
    font-weight: 600;
    box-shadow: 0 1px 3px rgba(0, 0, 0, 0.05);
}

.safe-status {
    color: #059669;
    text-align: center;
    font-size: 14px;
    font-weight: 600;
    padding: 20px;
    background: #d1fae5;
    border-radius: 8px;
    border: 1px solid #a7f3d0;
}

/* Emergency Contact Card */
.contact-card {
    background: #ffffff;
    padding: 16px;
    border-radius: 8px;
    border: 1px solid #e2e8f0;
    margin: 10px 0;
    transition: all 0.2s ease;
}

.contact-card:hover {
    border-color: #cbd5e0;
    box-shadow: 0 2px 8px rgba(0,0,0,0.06);
}

.contact-name {
    color: #0f172a !important;
    font-weight: 700;
    font-size: 15px;
    margin-bottom: 8px;
}

.contact-phone {
    color: #3b82f6 !important;
    font-size: 17px;
    font-weight: 700;
    font-family: 'Courier New', monospace;
    margin: 6px 0;
    cursor: pointer;
    text-decoration: none;
    display: inline-block;
    transition: all 0.2s ease;
}

.contact-phone a {
    color: #3b82f6 !important;
    text-decoration: none;
    font-weight: 700;
}

.contact-phone:hover,
.contact-phone a:hover {
    color: #2563eb !important;
    text-decoration: underline;
    transform: scale(1.02);
}

.contact-relation {
    color: #475569 !important;
    font-size: 11px;
    text-transform: uppercase;
    letter-spacing: 1px;
    font-weight: 700;
}

/* Data Grid - Professional Table Style */
.data-grid {
    background: #ffffff;
    border-radius: 8px;
    padding: 15px;
    border: 1px solid #e2e8f0;
    margin: 10px 0;
}

.data-row {
    display: flex;
    justify-content: space-between;
    padding: 10px 0;
    border-bottom: 1px solid #f7fafc;
}

.data-label {
    color: #0f172a !important;
    font-size: 15px;
    font-weight: 800;
    text-transform: uppercase;
    letter-spacing: 0.5px;
}

.data-value {
    color: #0f172a !important;
    font-size: 17px;
    font-weight: 900;
}

/* Camera and Map Containers */
.media-container {
    border-radius: 8px;
    overflow: hidden;
    border: 1px solid #e2e8f0;
    box-shadow: 0 2px 8px rgba(0,0,0,0.06);
    background: #ffffff;
    padding: 3px;
}

/* AI Report Box */
.ai-report-box {
    background: #ffffff;
    padding: 30px;
    border-radius: 8px;
    border: 1px solid #e2e8f0;
    box-shadow: 0 2px 8px rgba(0,0,0,0.06);
    margin-bottom: 25px;
}

.ai-report-title {
    color: #2d3748;
    font-size: 20px;
    font-weight: 600;
    margin-bottom: 20px;
    padding-bottom: 15px;
    border-bottom: 2px solid #e2e8f0;
}

.ai-report-content {
    color: #4a5568;
    line-height: 1.8;
    font-size: 14px;
}

/* Sidebar Styling - Premium with Light Background */
section[data-testid="stSidebar"] {
    background: #ffffff;
    border-right: 1px solid #e2e8f0;
}

section[data-testid="stSidebar"] > div {
    background: #ffffff;
}

/* Sidebar Text - All Dark and Readable */
section[data-testid="stSidebar"] p,
section[data-testid="stSidebar"] span,
section[data-testid="stSidebar"] div,
section[data-testid="stSidebar"] label {
    color: #0f172a !important;
    font-weight: 600 !important;
}

/* Sidebar Inputs - Light Background, Dark Text */
section[data-testid="stSidebar"] input,
section[data-testid="stSidebar"] select {
    background: #ffffff !important;
    color: #0f172a !important;
    border: 1px solid #cbd5e0 !important;
    font-weight: 600 !important;
    font-size: 14px !important;
}

section[data-testid="stSidebar"] input::placeholder {
    color: #64748b !important;
    font-weight: 500 !important;
}

/* Sidebar Selectbox - Fix Dark Text on Dark Background */
section[data-testid="stSidebar"] select option {
    background: #ffffff !important;
    color: #0f172a !important;
    font-weight: 600 !important;
}

/* Dropdown Options - Ensure Light Background */
.stSelectbox div[data-baseweb="select"] > div {
    background: #ffffff !important;
}

.stSelectbox [role="listbox"] {
    background: #ffffff !important;
}

.stSelectbox [role="option"] {
    background: #ffffff !important;
    color: #0f172a !important;
    font-weight: 600 !important;
}

.stSelectbox [role="option"]:hover {
    background: #f8fafc !important;
    color: #0f172a !important;
}

/* Expander in Sidebar */
section[data-testid="stSidebar"] .streamlit-expanderHeader {
    background: #f8fafc !important;
    color: #0f172a !important;
    font-weight: 700 !important;
}

/* Button Styling - Professional */
.stButton>button {
    background: #3498db;
    color: white;
    border: none;
    border-radius: 6px;
    padding: 10px 20px;
    font-weight: 600;
    font-size: 14px;
    transition: all 0.2s ease;
    box-shadow: 0 1px 3px rgba(52, 152, 219, 0.2);
}

.stButton>button:hover {
    background: #2980b9;
    box-shadow: 0 2px 6px rgba(52, 152, 219, 0.3);
}

/* Metric Cards - Streamlit Native */
div[data-testid="metric-container"] {
    background: #ffffff;
    border: 1px solid #e2e8f0;
    border-radius: 8px;
    padding: 16px;
    box-shadow: 0 1px 3px rgba(0,0,0,0.04);
}

div[data-testid="metric-container"] label {
    color: #718096;
    font-size: 12px;
    font-weight: 600;
}

div[data-testid="metric-container"] [data-testid="stMetricValue"] {
    color: #2d3748;
    font-size: 24px;
    font-weight: 700;
}

/* Progress Indicator */
.progress-bar {
    height: 6px;
    background: #e2e8f0;
    border-radius: 3px;
    overflow: hidden;
    margin-top: 8px;
}

.progress-fill {
    height: 100%;
    background: #3498db;
    transition: width 0.3s ease;
}

/* Scrollbar Styling */
::-webkit-scrollbar {
    width: 8px;
    height: 8px;
}

::-webkit-scrollbar-track {
    background: #f1f5f9;
}

::-webkit-scrollbar-thumb {
    background: #cbd5e0;
    border-radius: 4px;
}

::-webkit-scrollbar-thumb:hover {
    background: #a0aec0;
}

/* Remove default Streamlit padding */
.block-container {
    padding-top: 2rem;
    padding-bottom: 2rem;
}

/* Expander Styling */
.streamlit-expanderHeader {
    background: #ffffff;
    border: 1px solid #e2e8f0;
    border-radius: 6px;
    font-weight: 600;
    color: #2d3748;
}

/* Input Styling */
.stTextInput>div>div>input {
    border-radius: 6px;
    border: 1px solid #e2e8f0;
}

.stSelectbox>div>div>select {
    border-radius: 6px;
    border: 1px solid #e2e8f0;
}
</style>
""", unsafe_allow_html=True)

if "view_mode" not in st.session_state: st.session_state["view_mode"] = "home"
if "trip_data" not in st.session_state: st.session_state["trip_data"] = []
if "trip_history" not in st.session_state: st.session_state["trip_history"] = []
if "ai_report" not in st.session_state: st.session_state["ai_report"] = ""
if "emergency_log" not in st.session_state: st.session_state["emergency_log"] = []
if "vehicle_sim" not in st.session_state: st.session_state["vehicle_sim"] = VehicleDynamicsSimulator()
if "audio_sim" not in st.session_state: st.session_state["audio_sim"] = AudioStressAnalyzer()
if "facial_stress_detector" not in st.session_state: st.session_state["facial_stress_detector"] = FacialStressDetector()

# Emergency Contacts Management
if "emergency_contacts" not in st.session_state:
    st.session_state["emergency_contacts"] = [
        {"name": "Emergency Services", "phone": "108", "relation": "Emergency"},
        {"name": "Police", "phone": "100", "relation": "Emergency"},
        {"name": "Ambulance", "phone": "102", "relation": "Emergency"}
    ]

# Sidebar Controls
st.sidebar.markdown("<h2 style='color: #2d3748; font-weight: 600; font-size: 18px; margin-bottom: 20px;'>🎛️ Dashboard Controls</h2>", unsafe_allow_html=True)

if st.sidebar.button("🏠 HOME", use_container_width=True):
    st.session_state["view_mode"] = "home"
    st.rerun()

if st.sidebar.button("▶ START TRIP", use_container_width=True):
    st.session_state["view_mode"] = "dashboard"
    st.session_state["trip_data"] = []
    st.session_state["ai_report"] = ""
    run = True
else: run = False

if st.sidebar.button("⏹ END TRIP & REPORT", use_container_width=True):
    st.session_state["view_mode"] = "report"
    run = False

st.sidebar.markdown("---")

# Emergency Contacts Section
st.sidebar.markdown("<h3 style='color: #2d3748; font-weight: 600; font-size: 16px;'>📞 Emergency Contacts</h3>", unsafe_allow_html=True)

# Add new contact
with st.sidebar.expander("➕ Add New Contact"):
    new_name = st.text_input("Name", key="new_contact_name")
    new_phone = st.text_input("Phone Number", key="new_contact_phone")
    new_relation = st.selectbox("Relation", ["Family", "Friend", "Doctor", "Other"], key="new_contact_relation")
    
    if st.button("Add Contact", use_container_width=True):
        if new_name and new_phone:
            st.session_state["emergency_contacts"].append({
                "name": new_name,
                "phone": new_phone,
                "relation": new_relation
            })
            st.success(f"✅ Added {new_name}")
            st.rerun()

# Display contacts
for idx, contact in enumerate(st.session_state["emergency_contacts"]):
    with st.sidebar.container():
        st.markdown(f"""
        <div class='contact-card'>
            <div class='contact-relation'>{contact['relation']}</div>
            <div class='contact-name'>{contact['name']}</div>
            <div class='contact-phone'>📞 <a href="tel:{contact['phone']}">{contact['phone']}</a></div>
        </div>
        """, unsafe_allow_html=True)
        
        # Delete button for non-emergency contacts
        if idx >= 3:  # Don't allow deleting first 3 emergency services
            if st.button(f"🗑️ Remove", key=f"del_{idx}", use_container_width=True):
                st.session_state["emergency_contacts"].pop(idx)
                st.rerun()

st.sidebar.markdown("---")

# Settings
st.sidebar.markdown("<h3 style='color: #2d3748; font-weight: 600; font-size: 16px;'>⚙️ Settings</h3>", unsafe_allow_html=True)
sensitivity = st.sidebar.slider("Eye Detection Sensitivity", 5, 25, 12, help="Higher = More Strict")
force_drowsy = st.sidebar.checkbox("🧪 Test: Force Drowsy Mode")
force_critical = st.sidebar.checkbox("🚨 Test: Force Emergency Mode")

# ==========================================
# 9. HOME PAGE VIEW (Trip History Dashboard)
# ==========================================
if st.session_state["view_mode"] == "home":
    # Main Header
    st.markdown("""
    <div class='main-header'>
        <div class='main-title'>🏠 Driver Safety System - Home</div>
        <div class='sub-title'>Trip History & Analytics Dashboard</div>
    </div>
    """, unsafe_allow_html=True)
    
    # Welcome Section
    st.markdown("""
    <div style='background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); padding: 40px; border-radius: 16px; margin: 20px 0; color: white;'>
        <h1 style='color: white; font-size: 36px; margin-bottom: 16px;'>Welcome to Your Safety Dashboard</h1>
        <p style='font-size: 18px; opacity: 0.95;'>Monitor your driving patterns, track safety metrics, and improve your driving behavior with AI-powered insights.</p>
    </div>
    """, unsafe_allow_html=True)
    
    # Quick Stats
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.markdown(f"""
        <div class='kpi-card'>
            <div class='kpi-icon'>🚗</div>
            <div class='kpi-label'>Total Trips</div>
            <div class='kpi-value status-safe'>{len(st.session_state['trip_history'])}</div>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        avg_risk = sum([t.get('avg_risk', 0) for t in st.session_state['trip_history']]) / max(len(st.session_state['trip_history']), 1)
        risk_color = "status-danger" if avg_risk > 60 else "status-warning" if avg_risk > 40 else "status-safe"
        st.markdown(f"""
        <div class='kpi-card'>
            <div class='kpi-icon'>⚠️</div>
            <div class='kpi-label'>Avg Risk Score</div>
            <div class='kpi-value {risk_color}'>{avg_risk:.0f}%</div>
        </div>
        """, unsafe_allow_html=True)
    
    with col3:
        total_emergencies = sum([len(t.get('emergencies', [])) for t in st.session_state['trip_history']])
        st.markdown(f"""
        <div class='kpi-card'>
            <div class='kpi-icon'>🚨</div>
            <div class='kpi-label'>Emergency Events</div>
            <div class='kpi-value status-danger'>{total_emergencies}</div>
        </div>
        """, unsafe_allow_html=True)
    
    with col4:
        total_stress = sum([t.get('stress_events', 0) for t in st.session_state['trip_history']])
        st.markdown(f"""
        <div class='kpi-card'>
            <div class='kpi-icon'>😰</div>
            <div class='kpi-label'>Stress Events</div>
            <div class='kpi-value status-warning'>{total_stress}</div>
        </div>
        """, unsafe_allow_html=True)
    
    st.markdown("---")
    
    # Trip History
    st.markdown("<div class='section-header'><span class='section-icon'>📜</span>Recent Trip History</div>", unsafe_allow_html=True)
    
    if len(st.session_state['trip_history']) == 0:
        st.info("📭 No trips recorded yet. Click 'START TRIP' to begin your first journey!")
    else:
        for idx, trip in enumerate(reversed(st.session_state['trip_history'][-10:])):
            trip_num = len(st.session_state['trip_history']) - idx
            risk_color = "🔴" if trip.get('avg_risk', 0) > 60 else "🟡" if trip.get('avg_risk', 0) > 40 else "🟢"
            
            with st.expander(f"🚗 Trip #{trip_num} - {trip.get('date', 'Unknown')} {risk_color}"):
                col1, col2, col3 = st.columns(3)
                
                with col1:
                    st.metric("Duration", f"{trip.get('duration', 0)}s")
                    st.metric("Avg Speed", f"{trip.get('avg_speed', 0)} km/h")
                
                with col2:
                    st.metric("Avg Risk", f"{trip.get('avg_risk', 0):.0f}%")
                    st.metric("Max Risk", f"{trip.get('max_risk', 0):.0f}%")
                
                with col3:
                    st.metric("Drowsy Events", trip.get('drowsy_count', 0))
                    st.metric("Stress Events", trip.get('stress_events', 0))
                
                if trip.get('emergencies', []):
                    st.warning(f"⚠️ {len(trip['emergencies'])} Emergency Event(s) Detected")
    
    # Quick Actions
    st.markdown("---")
    st.markdown("<div class='section-header'><span class='section-icon'>🎯</span>Quick Actions</div>", unsafe_allow_html=True)
    
    col1, col2, col3 = st.columns(3)
    with col1:
        if st.button("▶️ Start New Trip", use_container_width=True, type="primary"):
            st.session_state["view_mode"] = "dashboard"
            st.session_state["trip_data"] = []
            st.rerun()
    
    with col2:
        if st.button("📊 View Last Report", use_container_width=True):
            if st.session_state.get("ai_report"):
                st.session_state["view_mode"] = "report"
                st.rerun()
            else:
                st.warning("No report available. Complete a trip first!")
    
    with col3:
        if st.button("🗑️ Clear History", use_container_width=True):
            st.session_state["trip_history"] = []
            st.rerun()

# ==========================================
# 10. DASHBOARD VIEW (Live Monitoring)
# ==========================================
elif st.session_state["view_mode"] == "dashboard":
    # Main Header
    st.markdown("""
    <div class='main-header'>
        <div class='main-title'>🚗 Fusion-Based Accident Predictor</div>
        <div class='sub-title'>Advanced Multi-Modal AI System for Proactive Vehicle Safety</div>
    </div>
    """, unsafe_allow_html=True)

    # EMERGENCY BANNER PLACEHOLDER
    emergency_banner = st.empty()

    # TOP ROW: Primary KPIs
    st.markdown("<div class='section-header'><span class='section-icon'>📊</span>Real-Time Risk Metrics</div>", unsafe_allow_html=True)
    c1, c2, c3, c4 = st.columns(4)
    risk_box = c1.empty()
    speed_box = c2.empty()
    eye_box = c3.empty()
    loc_box = c4.empty()

    # SECOND ROW: Vehicle & Audio Metrics
    st.markdown("<div class='section-header'><span class='section-icon'>🚙</span>Vehicle & Environmental Data</div>", unsafe_allow_html=True)
    v1, v2, v3, v4 = st.columns(4)
    steering_box = v1.empty()
    brake_box = v2.empty()
    audio_box = v3.empty()
    weather_box = v4.empty()

    # MIDDLE ROW: Camera & GPS Map
    st.markdown("<div class='section-header'><span class='section-icon'>📹</span>Live Monitoring Systems</div>", unsafe_allow_html=True)
    cam_col, map_col = st.columns([1, 1])
    with cam_col:
        st.markdown("**Driver Vision Feed**")
        cam_place = st.empty()
    with map_col:
        st.markdown("**GPS & Accident Zones**")
        map_place = st.empty()
    
    # BOTTOM ROW: Risk Factors
    st.markdown("<div class='section-header'><span class='section-icon'>⚠️</span>Risk Factor Analysis</div>", unsafe_allow_html=True)
    risk_factors_box = st.empty()

    if run:
        if not voice_enabled:
            st.sidebar.error("Voice Alert Disabled. Install pyttsx3.")

        cap = cv2.VideoCapture(0)
        step = 0
        last_sim_update = time.time()
        last_voice_alert = time.time() - 10 # Cooldown timer for voice
        
        drowsy_start_time = None
        current_eye_state = "ALERT"
        user_history = get_user_history()
        sim_speed = 40
        
        # Initial GPS Coordinates (Near first accident zone)
        current_lat, current_lon = 12.9165, 79.1325
        
        # Initialize simulators
        vehicle_sim = st.session_state["vehicle_sim"]
        audio_sim = st.session_state["audio_sim"]
        
        emergency_triggered = False 

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret: break

            # --- A. OPENCV VISION ---
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            faces = face_cascade.detectMultiScale(gray, 1.3, 5) if face_cascade is not None else ()
            eyes_detected = 0
            
            if len(faces) > 0:
                for (x,y,w,h) in faces:
                    cv2.rectangle(rgb, (x,y), (x+w,y+h), (100,100,100), 2)
                    roi_gray = gray[y:y+int(h/2), x:x+w] # Top half of face
                    eyes = eye_cascade.detectMultiScale(roi_gray, 1.1, sensitivity) if eye_cascade is not None else ()
                    eyes_detected = len(eyes)
                    for (ex,ey,ew,eh) in eyes:
                        cv2.rectangle(rgb[y:y+int(h/2), x:x+w], (ex,ey), (ex+ew,ey+eh), (0,255,0), 2)

            # --- B. BUFFER LOGIC ---
            if len(faces) > 0 and eyes_detected < 2:
                if drowsy_start_time is None: drowsy_start_time = time.time()
                elif time.time() - drowsy_start_time > 1.5: current_eye_state = "DROWSY"
            else:
                drowsy_start_time = None
                current_eye_state = "ALERT"

            if force_drowsy: current_eye_state = "DROWSY"

            # --- C. SIMULATION (Speed & GPS) ---
            if time.time() - last_sim_update > 1.5:
                step += 1
                # Simulate GPS Movement (move through accident zones)
                current_lat += random.uniform(-0.001, 0.001)
                current_lon += random.uniform(-0.001, 0.001)
                
                # Simulate Speed changes
                if step < 10: sim_speed = random.randint(35, 50)
                elif step < 20: sim_speed = random.randint(75, 95)  # High speed phase
                else: sim_speed = random.randint(25, 40)
                last_sim_update = time.time()

            # --- D. GET GPS CONTEXT ---
            gps_context = get_gps_context(current_lat, current_lon, sim_speed)
            
            # --- E. VEHICLE DYNAMICS ---
            vehicle_dynamics = vehicle_sim.update(sim_speed, 50 if current_eye_state == "DROWSY" else 20)
            
            # --- F. AUDIO STRESS ANALYSIS ---
            audio_stress = audio_sim.analyze(50 if current_eye_state == "DROWSY" else 20)
            
            # --- G. FACIAL STRESS & EMOTION DETECTION (NEW) ---
            facial_stress_detector = st.session_state["facial_stress_detector"]
            facial_stress = facial_stress_detector.analyze_facial_stress(
                face_detected=True,
                eye_state=current_eye_state,
                risk_level=50 if current_eye_state == "DROWSY" else 20
            )
            
            # --- H. DRIVER STATE (Enhanced with Stress) ---
            driver_state = {
                "eye_state": current_eye_state,
                "facial_tension": facial_stress["facial_tension"],
                "facial_stress": facial_stress,
                "stress_state": facial_stress["stress_state"],
                "emotion": facial_stress["emotion"]
            }

            # --- H. MULTI-MODAL FUSION ENGINE ---
            risk_label, risk_val, risk_color, risk_factors = calculate_fusion_risk(
                driver_state, vehicle_dynamics, audio_stress, gps_context, user_history
            )
            if force_critical: 
                risk_val, risk_label, risk_color = 95, "CRITICAL", "bad"
                risk_factors = ["FORCED TEST MODE - CRITICAL"]

            # --- I. AI VOICE ALERTS & EMERGENCY CALLING ---
            if risk_val > 75: # CRITICAL
                # Get emergency contacts for display
                contacts_list = st.session_state["emergency_contacts"]
                contacts_display = "<br>".join([f"📞 {c['name']}: {c['phone']}" for c in contacts_list[:5]])
                
                emergency_banner.markdown(f"""
                <div class='emergency-box'>
                    🚨 <strong>CRITICAL ACCIDENT RISK DETECTED!</strong><br><br>
                    <strong>EMERGENCY PROTOCOL ACTIVATED</strong><br><br>
                    📞 Calling Emergency Services & Contacts:<br>
                    {contacts_display}<br><br>
                    🚗 Auto-Brakes Applied | 💺 Seat Belts Tightened<br>
                    📍 GPS Location: {gps_context['lat']:.4f}, {gps_context['lon']:.4f}<br>
                    🚨 Hazard Lights Activated
                </div>
                """, unsafe_allow_html=True)
                
                # Trigger emergency call (once per critical event)
                if not emergency_triggered:
                    emergency_data = trigger_emergency_call(gps_context, risk_factors, contacts_list)
                    st.session_state["emergency_log"].append(emergency_data)
                    emergency_triggered = True
                
                # Voice Alert (Every 7 seconds to not overlap)
                if time.time() - last_voice_alert > 7:
                    speak_alert("Critical Warning. Accident Imminent. Emergency services and contacts have been notified. Please pull over safely.")
                    last_voice_alert = time.time()

            elif risk_val > 50: # HIGH
                emergency_banner.markdown("""
                <div class='warning-box'>
                    ⚠️ <strong>HIGH RISK WARNING</strong><br>
                    Multiple Risk Factors Detected - Please Reduce Speed and Stay Alert
                </div>
                """, unsafe_allow_html=True)
                emergency_triggered = False  # Reset for next critical event
                
                if time.time() - last_voice_alert > 8:
                    speak_alert("High Risk Warning. Please slow down and stay alert.")
                    last_voice_alert = time.time()
            else:
                emergency_banner.empty()
                emergency_triggered = False
            
            # --- ADDITIONAL VOICE ALERTS FOR STRESS & SHOUTING ---
            if facial_stress["stress_state"] == "RAGE" and time.time() - last_voice_alert > 10:
                speak_alert("Driver stress level critical. Please take deep breaths and stay calm.")
                last_voice_alert = time.time()
            
            if audio_stress["shouting_detected"] and audio_stress["shout_count"] > 2 and time.time() - last_voice_alert > 12:
                speak_alert("Elevated voice detected. Please remain calm for safe driving.")
                last_voice_alert = time.time()

            # --- J. UI DASHBOARD UPDATES WITH DETAILED METRICS ---
            # Primary KPIs with detailed information
            risk_box.markdown(f"""
            <div class='kpi-card'>
                <div class='kpi-icon'>🎯</div>
                <div class='kpi-label'>Fusion Risk Score</div>
                <div class='kpi-value {"status-danger" if risk_val > 75 else "status-warning" if risk_val > 50 else "status-safe"}'>{risk_val}%</div>
                <div class='kpi-sublabel'>
                    <span class='{"badge-danger" if risk_val > 75 else "badge-warning" if risk_val > 50 else "badge-safe"}'>{risk_label}</span>
                </div>
                <div class='data-grid' style='margin-top: 12px;'>
                    <div class='data-row'>
                        <span class='data-label'>Threshold</span>
                        <span class='data-value'>75% Critical</span>
                    </div>
                    <div class='data-row' style='border: none;'>
                        <span class='data-label'>Status</span>
                        <span class='data-value'>{"🔴 Alert" if risk_val > 75 else "🟡 Monitor" if risk_val > 50 else "🟢 Safe"}</span>
                    </div>
                </div>
            </div>
            """, unsafe_allow_html=True)
            
            speed_color = "status-danger" if sim_speed > gps_context["speed_limit"] else "status-safe"
            speed_box.markdown(f"""
            <div class='kpi-card'>
                <div class='kpi-icon'>🚗</div>
                <div class='kpi-label'>Vehicle Speed</div>
                <div class='kpi-value {speed_color}'>{sim_speed} <span style='font-size: 18px;'>km/h</span></div>
                <div class='kpi-sublabel'>
                    Speed Limit: {gps_context['speed_limit']} km/h
                </div>
                <div class='data-grid' style='margin-top: 12px;'>
                    <div class='data-row'>
                        <span class='data-label'>Avg Speed</span>
                        <span class='data-value'>{int(sum(d.get("speed", 0) for d in st.session_state["trip_data"][-10:]) / max(len(st.session_state["trip_data"][-10:]), 1))} km/h</span>
                    </div>
                    <div class='data-row' style='border: none;'>
                        <span class='data-label'>Violations</span>
                        <span class='data-value'>{sum(1 for d in st.session_state["trip_data"] if d.get("speed", 0) > 80)}</span>
                    </div>
                </div>
            </div>
            """, unsafe_allow_html=True)
            
            e_col = "status-danger" if current_eye_state == "DROWSY" else "status-safe"
            stress_col = "status-danger" if facial_stress["stress_state"] == "RAGE" else "status-warning" if facial_stress["stress_state"] == "STRESSED" else "status-safe"
            stress_emoji = "😡" if facial_stress["stress_state"] == "RAGE" else "😰" if facial_stress["stress_state"] == "STRESSED" else "😌"
            
            with eye_box.container():
                st.markdown(f"""
                <div class='kpi-card'>
                    <div class='kpi-icon'>👁️</div>
                    <div class='kpi-label'>DRIVER MONITORING</div>
                    <div class='kpi-value {e_col}'>{current_eye_state}</div>
                    <div class='kpi-sublabel'>
                        <span class='{"badge-danger" if facial_stress["stress_state"] == "RAGE" else "badge-warning" if facial_stress["stress_state"] == "STRESSED" else "badge-safe"}' style='font-size: 16px; padding: 8px 16px;'>
                            {stress_emoji} {facial_stress["stress_state"]} - {facial_stress["emotion"]}
                        </span>
                    </div>
                </div>
                """, unsafe_allow_html=True)
                
                # Use Streamlit metrics instead of HTML
                col1, col2 = st.columns(2)
                with col1:
                    st.metric("Stress Level", f"{facial_stress['stress_level']:.0%}")
                    st.metric("Jaw Clench", "⚠️ YES" if facial_stress["jaw_clench"] else "✓ No")
                    st.metric("Drowsy Events", sum(1 for d in st.session_state["trip_data"] if d.get("eye_state") == "DROWSY"))
                with col2:
                    st.metric("Facial Tension", f"{driver_state['facial_tension']:.0%}")
                    st.metric("Eyebrow Furrow", "⚠️ YES" if facial_stress["eyebrow_furrow"] else "✓ No")
                    st.metric("Stress Duration", f"{facial_stress['stress_duration']} frames")
            
            zone_color = "status-danger" if gps_context["zone_risk"] == "High" else ("status-warning" if gps_context["zone_risk"] == "Medium" else "status-safe")
            loc_box.markdown(f"""
            <div class='kpi-card'>
                <div class='kpi-icon'>📍</div>
                <div class='kpi-label'>GPS Zone</div>
                <div class='kpi-value {zone_color}' style='font-size: 20px;'>{gps_context['zone_name']}</div>
                <div class='kpi-sublabel'>
                    <span class='{"badge-danger" if gps_context["zone_risk"] == "High" else "badge-warning" if gps_context["zone_risk"] == "Medium" else "badge-safe"}'>{gps_context['zone_risk']} Risk</span>
                </div>
                <div class='data-grid' style='margin-top: 12px;'>
                    <div class='data-row'>
                        <span class='data-label'>Coordinates</span>
                        <span class='data-value'>{gps_context['lat']:.4f}, {gps_context['lon']:.4f}</span>
                    </div>
                    <div class='data-row' style='border: none;'>
                        <span class='data-label'>Weather</span>
                        <span class='data-value'>{gps_context['weather_risk']}</span>
                    </div>
                </div>
            </div>
            """, unsafe_allow_html=True)
            
            # Vehicle & Audio Metrics with detailed data
            steer_color = "status-danger" if vehicle_dynamics["aggressive_steering"] else "status-safe"
            steering_box.markdown(f"""
            <div class='kpi-card'>
                <div class='kpi-icon'>🎮</div>
                <div class='kpi-label'>Steering Behavior</div>
                <div class='kpi-value {steer_color}'>{abs(vehicle_dynamics['steering_angle']):.1f}°</div>
                <div class='kpi-sublabel'>
                    {'⚠️ Aggressive' if vehicle_dynamics['aggressive_steering'] else '✓ Normal'}
                </div>
                <div class='data-grid' style='margin-top: 12px;'>
                    <div class='data-row'>
                        <span class='data-label'>Jerks Detected</span>
                        <span class='data-value'>{sum(1 for d in st.session_state["trip_data"] if d.get("steering", 0) > 10)}</span>
                    </div>
                    <div class='data-row' style='border: none;'>
                        <span class='data-label'>Smoothness</span>
                        <span class='data-value'>{"Low" if vehicle_dynamics['aggressive_steering'] else "High"}</span>
                    </div>
                </div>
            </div>
            """, unsafe_allow_html=True)
            
            brake_color = "status-danger" if vehicle_dynamics["harsh_braking"] else "status-safe"
            brake_box.markdown(f"""
            <div class='kpi-card'>
                <div class='kpi-icon'>🛑</div>
                <div class='kpi-label'>Braking Force</div>
                <div class='kpi-value {brake_color}'>{vehicle_dynamics['brake_force']:.0%}</div>
                <div class='kpi-sublabel'>
                    {'⚠️ Harsh Braking' if vehicle_dynamics['harsh_braking'] else '✓ Normal'}
                </div>
                <div class='data-grid' style='margin-top: 12px;'>
                    <div class='data-row'>
                        <span class='data-label'>Harsh Events</span>
                        <span class='data-value'>{sum(1 for d in st.session_state["trip_data"] if d.get("brake", 0) > 0.7)}</span>
                    </div>
                    <div class='data-row' style='border: none;'>
                        <span class='data-label'>Avg Force</span>
                        <span class='data-value'>{int(sum(d.get("brake", 0) for d in st.session_state["trip_data"][-10:]) / max(len(st.session_state["trip_data"][-10:]), 1) * 100)}%</span>
                    </div>
                </div>
            </div>
            """, unsafe_allow_html=True)
            
            audio_color = "status-danger" if audio_stress["stress_level"] == "HIGH" else ("status-warning" if audio_stress["stress_level"] == "MEDIUM" else "status-safe")
            
            with audio_box.container():
                st.markdown(f"""
                <div class='kpi-card'>
                    <div class='kpi-icon'>🎤</div>
                    <div class='kpi-label'>VOICE & AUDIO ANALYSIS</div>
                    <div class='kpi-value {audio_color}'>{audio_stress['stress_level']}</div>
                    <div class='kpi-sublabel'>
                        <span class='{"badge-danger" if audio_stress["shouting_detected"] else "badge-safe"}' style='font-size: 16px; padding: 8px 16px;'>
                            {"🔊 SHOUTING!" if audio_stress['shouting_detected'] else "🔇 Calm Voice"}
                        </span>
                    </div>
                </div>
                """, unsafe_allow_html=True)
                
                # Use Streamlit metrics instead of HTML
                col1, col2 = st.columns(2)
                with col1:
                    st.metric("Stress Value", f"{audio_stress['stress_value']:.0%}")
                    st.metric("Voice Pitch", f"{audio_stress['voice_pitch']:.0f} Hz")
                    st.metric("Shout Count", f"⚠️ {audio_stress['shout_count']}")
                with col2:
                    st.metric("Cabin Noise", f"{audio_stress['cabin_noise']:.0f} dB")
                    st.metric("Agitation", audio_stress['voice_agitation'])
            
            weather_color = "status-danger" if gps_context["weather_risk"] == "Poor" else "status-safe"
            weather_box.markdown(f"""
            <div class='kpi-card'>
                <div class='kpi-icon'>🌤️</div>
                <div class='kpi-label'>Weather Conditions</div>
                <div class='kpi-value {weather_color}' style='font-size: 22px;'>{gps_context['weather_risk']}</div>
                <div class='kpi-sublabel'>
                    Road Conditions
                </div>
                <div class='data-grid' style='margin-top: 12px;'>
                    <div class='data-row'>
                        <span class='data-label'>Visibility</span>
                        <span class='data-value'>{"Low" if gps_context['weather_risk'] == "Poor" else "Good"}</span>
                    </div>
                    <div class='data-row' style='border: none;'>
                        <span class='data-label'>Road Grip</span>
                        <span class='data-value'>{"Reduced" if gps_context['weather_risk'] == "Poor" else "Normal"}</span>
                    </div>
                </div>
            </div>
            """, unsafe_allow_html=True)
            
            # Camera Feed
            cv2.putText(rgb, f"STATUS: {current_eye_state}", (20, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255,0,0) if current_eye_state == "DROWSY" else (0,255,0), 2)
            cv2.putText(rgb, f"RISK: {risk_val}% ({risk_label})", (20, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255,0,0) if risk_val > 50 else (0,255,0), 2)
            cam_place.image(rgb, channels="RGB", use_container_width=True)
            
            # GPS Map with Accident Zones
            map_data = pd.DataFrame({
                'lat': [current_lat] + [z["lat"] for z in ACCIDENT_PRONE_ZONES],
                'lon': [current_lon] + [z["lon"] for z in ACCIDENT_PRONE_ZONES],
                'type': ['Current'] + [z["risk"] for z in ACCIDENT_PRONE_ZONES]
            })
            map_place.map(map_data, zoom=13, use_container_width=True)
            
            # Risk Factors Display with detailed information
            if risk_factors:
                factors_html = "<div class='risk-factors-box'>"
                factors_html += "<div class='risk-factor-header'>⚠️ Active Risk Factors Detected</div>"
                for idx, factor in enumerate(risk_factors, 1):
                    factors_html += f"<div class='risk-factor-item'><strong>{idx}.</strong> {factor}</div>"
                factors_html += f"""
                <div style='margin-top: 15px; padding-top: 15px; border-top: 1px solid #e2e8f0;'>
                    <div class='data-row'>
                        <span class='data-label'>Total Factors</span>
                        <span class='data-value'>{len(risk_factors)}</span>
                    </div>
                    <div class='data-row'>
                        <span class='data-label'>Risk Level</span>
                        <span class='data-value'>{risk_label}</span>
                    </div>
                    <div class='data-row' style='border: none;'>
                        <span class='data-label'>Action Required</span>
                        <span class='data-value'>{"Immediate" if risk_val > 75 else "Monitor" if risk_val > 50 else "None"}</span>
                    </div>
                </div>
                """
                factors_html += "</div>"
                risk_factors_box.markdown(factors_html, unsafe_allow_html=True)
            else:
                risk_factors_box.markdown(f"""
                <div class='safe-status'>
                    ✓ All Systems Normal - Safe Driving Conditions
                    <div style='margin-top: 10px; font-size: 12px; opacity: 0.8;'>
                        No risk factors detected • Trip duration: {len(st.session_state["trip_data"])}s
                    </div>
                </div>
                """, unsafe_allow_html=True)
            
            # Data Logging
            st.session_state["trip_data"].append({
                "time": datetime.now().strftime("%H:%M:%S"),
                "speed": sim_speed,
                "risk": risk_val,
                "status": risk_label,
                "lat": current_lat,
                "lon": current_lon,
                "zone": gps_context["zone_name"],
                "eye_state": current_eye_state,
                "stress_state": facial_stress["stress_state"],
                "emotion": facial_stress["emotion"],
                "facial_tension": facial_stress["facial_tension"],
                "steering": vehicle_dynamics["steering_angle"],
                "brake": vehicle_dynamics["brake_force"],
                "audio_stress": audio_stress["stress_level"],
                "shouting": audio_stress["shouting_detected"],
                "shout_count": audio_stress["shout_count"]
            })
            
            # Camera
            cv2.putText(rgb, f"STATUS: {current_eye_state}", (20, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255,0,0) if current_eye_state == "DROWSY" else (0,255,0), 2)
            cam_place.image(rgb, channels="RGB", use_container_width=True)
            
            # GPS Map Update
            map_data = pd.DataFrame({'lat': [current_lat], 'lon': [current_lon]})
            map_place.map(map_data, zoom=14, use_container_width=True)
            
            # Balanced delay: Fast enough for smooth camera, slow enough to prevent flickering
            time.sleep(0.1)  # 10 FPS - smooth camera feed with stable dashboard

        cap.release()

# ==========================================
# 11. POST-TRIP AI REPORT VIEW
# ==========================================
elif st.session_state["view_mode"] == "report":
    # Header
    st.markdown("""
    <div class='main-header'>
        <div class='main-title'>📊 Trip Analysis & Safety Report</div>
        <div class='sub-title'>Comprehensive AI-Powered Driving Assessment</div>
    </div>
    """, unsafe_allow_html=True)
    
    if not st.session_state["trip_data"]:
        st.warning("⚠️ No trip data recorded. Go back and press Start Trip.")
    else:
        # Emergency Log Display
        if st.session_state["emergency_log"]:
            st.markdown("<div class='section-header'>🚨 Emergency Events Log</div>", unsafe_allow_html=True)
            for event in st.session_state["emergency_log"]:
                st.markdown(f"""
                <div class='emergency-box' style='animation: none; text-align: left;'>
                    <strong>🚨 Emergency Triggered:</strong> {event['timestamp']}<br>
                    <strong>📍 Location:</strong> {event['location']} ({event['zone']})<br>
                    <strong>⚠️ Risk Factors:</strong> {', '.join(event['risk_factors'])}<br>
                    <strong>📞 Contacts Notified:</strong><br>
                    {'<br>'.join(['  • ' + c for c in event.get('contacts_notified', [])])}<br>
                    <strong>✅ Status:</strong> {event['status']}
                </div>
                """, unsafe_allow_html=True)
            st.markdown("<br>", unsafe_allow_html=True)
        
        # Generate AI Report
        if st.session_state["ai_report"] == "":
            with st.spinner("🤖 Gemini AI is analyzing your driving patterns..."):
                data = st.session_state["trip_data"]
                avg_risk = int(sum(d.get('risk', 0) for d in data) / len(data)) if data else 0
                max_speed = max(d.get('speed', 0) for d in data) if data else 0
                critical_events = sum(1 for d in data if d.get('status') == 'CRITICAL')
                high_risk_events = sum(1 for d in data if d.get('status') in ['CRITICAL', 'HIGH'])
                drowsy_count = sum(1 for d in data if d.get('eye_state') == 'DROWSY')
                harsh_brakes = sum(1 for d in data if d.get('brake', 0) > 0.7)
                
                prompt = f"""
                You are an AI Driving Safety Coach. Analyze this trip data:
                
                Trip Statistics:
                - Average Risk Score: {avg_risk}/100
                - Maximum Speed: {max_speed} km/h
                - Critical Events: {critical_events}
                - High Risk Events: {high_risk_events}
                - Drowsiness Detections: {drowsy_count}
                - Harsh Braking Events: {harsh_brakes}
                - Emergency Calls: {len(st.session_state["emergency_log"])}
                
                Provide:
                1. Overall Safety Verdict (Safe/Moderate/Unsafe)
                2. Key Risk Patterns Identified
                3. Specific Coaching Tips (2-3 actionable recommendations)
                4. Positive Behaviors (if any)
                
                Keep it professional, constructive, and concise.
                """
                st.session_state["ai_report"] = get_gemini_response(prompt)
                
                # Save trip to history
                trip_summary = {
                    'date': datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                    'duration': len(data),
                    'avg_speed': int(sum(d.get('speed', 0) for d in data) / len(data)) if data else 0,
                    'avg_risk': avg_risk,
                    'max_risk': max(d.get('risk', 0) for d in data) if data else 0,
                    'drowsy_count': drowsy_count,
                    'stress_events': sum(1 for d in data if d.get('stress_state') in ['STRESSED', 'RAGE']),
                    'emergencies': st.session_state["emergency_log"].copy()
                }
                st.session_state["trip_history"].append(trip_summary)
        
        # Display AI Report
        st.markdown("<div class='section-header'>🤖 AI Driving Safety Coach</div>", unsafe_allow_html=True)
        st.markdown(f"""
        <div class='ai-report-box'>
            <h3>🤖 AI Driving Safety Coach</h3>
            <div style='color: #cbd5e1; line-height: 1.8; font-size: 15px;'>
                {st.session_state["ai_report"].replace(chr(10), '<br>')}
            </div>
        </div>
        """, unsafe_allow_html=True)
        
        # Trip Statistics
        st.markdown("<div class='section-header'>📈 Trip Statistics</div>", unsafe_allow_html=True)
        col1, col2, col3, col4 = st.columns(4)
        data = st.session_state["trip_data"]
        
        with col1:
            avg_risk = int(sum(d.get('risk', 0) for d in data) / len(data)) if data else 0
            risk_color = "🔴" if avg_risk > 50 else "🟡" if avg_risk > 25 else "🟢"
            st.metric("Average Risk Score", f"{risk_color} {avg_risk}%")
        with col2:
            critical_events = sum(1 for d in data if d.get('status') == 'CRITICAL')
            st.metric("Critical Events", f"🚨 {critical_events}")
        with col3:
            max_speed = max(d.get('speed', 0) for d in data) if data else 0
            st.metric("Max Speed", f"🚗 {max_speed} km/h")
        with col4:
            drowsy_count = sum(1 for d in data if d.get('eye_state') == 'DROWSY')
            st.metric("Drowsiness Events", f"😴 {drowsy_count}")
        
        st.markdown("---")
        
        # Display GPS & Telemetry Log
        st.markdown("<div class='section-header'>📋 Detailed Telemetry Log</div>", unsafe_allow_html=True)
        df = pd.DataFrame(st.session_state["trip_data"])
        st.dataframe(df, use_container_width=True, height=400)
        
        # Download option
        col1, col2 = st.columns([3, 1])
        with col1:
            csv = df.to_csv(index=False)
            st.download_button(
                label="📥 Download Trip Data (CSV)",
                data=csv,
                file_name=f"trip_data_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                mime="text/csv",
                use_container_width=True
            )
        with col2:
            if st.button("⬅ Back to Dashboard", use_container_width=True):
                st.session_state["view_mode"] = "dashboard"
                st.rerun()