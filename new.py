import streamlit as st
import cv2
import pandas as pd
import numpy as np
import time
import random
from datetime import datetime

# ==========================================
# 1. SIMULATED AI (The "Mock" Engine)
# ==========================================
# This function mimics Google Gemini but runs offline.
# It analyzes your actual trip data and picks a smart response.

def get_mock_ai_response(avg_risk, max_speed, drowsy_count):
    if avg_risk > 50 or drowsy_count > 2:
        return f"""
        **🚨 CRITICAL SAFETY REPORT**
        
        **Safety Score:** {100 - avg_risk}/100 (POOR)
        
        **Analysis:**
        The driver demonstrated dangerous behavior. Drowsiness was detected {drowsy_count} times, and the vehicle reached high speeds ({max_speed} km/h) during these periods. This combination significantly increases the risk of a fatal collision.
        
        **Key Dangerous Moment:**
        At high speed, the driver's eyes were closed for over 1.5 seconds. This is a critical lapse in attention.
        
        **Recommendation:**
        Immediate rest is required. The driver should not continue operating the vehicle until they have slept for at least 20 minutes.
        """
    elif avg_risk > 20:
        return f"""
        **⚠️ CAUTIONARY SAFETY REPORT**
        
        **Safety Score:** {100 - avg_risk}/100 (AVERAGE)
        
        **Analysis:**
        The driver showed signs of mild distraction or fatigue. While no critical incidents occurred, the average risk level ({avg_risk}%) suggests the driver is not fully focused. Max speed was {max_speed} km/h.
        
        **Key Observation:**
        Occasional lane drift or delayed reactions were inferred from the telemetry.
        
        **Recommendation:**
        Consider taking a break soon. A caffeine stop or stretching break is advised to regain full alertness.
        """
    else:
        return f"""
        **✅ POSITIVE SAFETY REPORT**
        
        **Safety Score:** {100 - avg_risk}/100 (EXCELLENT)
        
        **Analysis:**
        The driver maintained excellent control of the vehicle. Drowsiness was not a factor, and speed was kept within safe limits (Max: {max_speed} km/h).
        
        **Key Observation:**
        Consistent attention to the road and smooth driving dynamics were observed.
        
        **Recommendation:**
        Keep up the good driving habits. No corrective action is needed.
        """

# ==========================================
# 2. OPENCV SETUP
# ==========================================
try:
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
    eye_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_eye.xml')
except:
    st.error("OpenCV Error. Please reinstall opencv-python.")
    st.stop()

# ==========================================
# 3. LOGIC FUNCTIONS
# ==========================================
def calculate_fusion_risk(speed, noise, location_risk, eye_state, history):
    risk_score = 0
    # Weights
    if eye_state == "DROWSY": risk_score += 45
    if speed > 80: risk_score += 25
    if location_risk == "High": risk_score += 20
    if history["risk_profile"] == "Aggressive": risk_score += 10
    
    # Verdict
    if risk_score > 75: return "CRITICAL", risk_score, "bad"
    elif risk_score > 40: return "HIGH", risk_score, "warn"
    elif risk_score > 20: return "MEDIUM", risk_score, "good"
    else: return "LOW", risk_score, "good"

def get_user_history():
    return {"risk_profile": "Aggressive", "common_faults": ["Speeding", "Late Braking"]}

# ==========================================
# 4. UI CONFIG
# ==========================================
st.set_page_config(page_title="Driver Safety Monitor", layout="wide")

st.markdown("""
<style>
.stApp { background-color: #0B0F14; }
.main-title { font-size: 36px; color: #E6EDF3; text-align: center; font-weight: 700; }
.sub-title { font-size: 14px; color: #8B949E; text-align: center; margin-bottom: 25px; }
.kpi-card { background: linear-gradient(145deg, #161b22, #0d1117); border-radius: 12px; padding: 15px; text-align: center; border: 1px solid #30363d; margin-bottom: 10px; }
.kpi-value { font-size: 28px; font-weight: bold; color: #E6EDF3; }
.kpi-label { font-size: 12px; color: #8B949E; text-transform: uppercase; }
.good { color: #00D084; } .warn { color: #FFB020; } .bad { color: #FF4D4F; }
div.stButton > button { width: 100%; background-color: #21262d; color: white; border: 1px solid #30363d; height: 45px; border-radius: 8px; }
</style>
""", unsafe_allow_html=True)

if "view_mode" not in st.session_state: st.session_state["view_mode"] = "dashboard"
if "trip_data" not in st.session_state: st.session_state["trip_data"] = []
if "ai_report" not in st.session_state: st.session_state["ai_report"] = ""

# Sidebar
st.sidebar.title("Control Panel")
if st.sidebar.button("▶ START SYSTEM"):
    st.session_state["view_mode"] = "dashboard"
    st.session_state["trip_data"] = []
    st.session_state["ai_report"] = ""
    run = True
else: run = False

if st.sidebar.button("⏹ STOP & REPORT"):
    st.session_state["view_mode"] = "report"
    run = False

st.sidebar.markdown("---")
st.sidebar.markdown("### 👁️ Calibration")
# IMPORTANT: Slider maps to minNeighbors
sensitivity = st.sidebar.slider("Eye Strictness (Higher = Stricter)", 5, 25, 12)
st.sidebar.caption("Increase if it sees 'Ghost Eyes'. Decrease if it never sees eyes.")
force_drowsy = st.sidebar.checkbox("Simulation: Force Drowsy")

# ==========================================
# 5. DASHBOARD
# ==========================================
if st.session_state["view_mode"] == "dashboard":
    st.markdown("<div class='main-title'>Fusion-Based Accident Predictor</div>", unsafe_allow_html=True)
    st.markdown("<div class='sub-title'>Enhanced OpenCV Vision | Fusion Engine | Intelligent Analysis</div>", unsafe_allow_html=True)

    c1, c2, c3, c4 = st.columns(4)
    with c1: 
        st.markdown("<div class='kpi-card'><div class='kpi-label'>Vehicle Speed</div>", unsafe_allow_html=True)
        speed_box = st.empty()
        st.markdown("</div>", unsafe_allow_html=True)
    with c2: 
        st.markdown("<div class='kpi-card'><div class='kpi-label'>Risk Profile</div>", unsafe_allow_html=True)
        hist_box = st.empty()
        st.markdown("</div>", unsafe_allow_html=True)
    with c3: 
        st.markdown("<div class='kpi-card'><div class='kpi-label'>Driver Status</div>", unsafe_allow_html=True)
        eye_box = st.empty()
        st.markdown("</div>", unsafe_allow_html=True)
    with c4: 
        st.markdown("<div class='kpi-card'><div class='kpi-label'>Fusion Risk</div>", unsafe_allow_html=True)
        risk_box = st.empty()
        st.markdown("</div>", unsafe_allow_html=True)

    cam_col, data_col = st.columns([2, 1])
    with cam_col: 
        cam_place = st.empty()
    with data_col:
        st.markdown("### Live Telemetry")
        event_log = st.empty()

    if run:
        cap = cv2.VideoCapture(0)
        step = 0
        last_sim_update = time.time()
        
        # Timer variables for drowsiness buffer
        drowsy_start_time = None
        current_eye_state = "ALERT"
        
        user_history = get_user_history()
        sim_speed, sim_loc = 0, "Low"

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret: break

            # A. ENHANCED OPENCV VISION
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            faces = face_cascade.detectMultiScale(gray, 1.3, 5)
            
            eyes_detected = 0
            
            if len(faces) > 0:
                for (x,y,w,h) in faces:
                    # Draw Face Box
                    cv2.rectangle(rgb, (x,y), (x+w,y+h), (100,100,100), 2)
                    
                    # SMART ROI: Only look for eyes in the TOP HALF of the face
                    roi_gray = gray[y : y + int(h/2), x : x + w]
                    roi_color = rgb[y : y + int(h/2), x : x + w]
                    
                    # Detect Eyes with User Sensitivity
                    eyes = eye_cascade.detectMultiScale(roi_gray, 1.1, sensitivity)
                    eyes_detected = len(eyes)
                    
                    for (ex,ey,ew,eh) in eyes:
                        cv2.rectangle(roi_color, (ex,ey), (ex+ew,ey+eh), (0,255,0), 2)

            # B. BUFFER LOGIC (Prevents Flickering)
            # If eyes < 2 (closed/missing)
            if len(faces) > 0 and eyes_detected < 2:
                if drowsy_start_time is None:
                    drowsy_start_time = time.time() # Start timer
                
                # Check if closed for > 1.5 seconds
                elif time.time() - drowsy_start_time > 1.5:
                    current_eye_state = "DROWSY"
            else:
                # Eyes found or no face -> Reset
                drowsy_start_time = None
                current_eye_state = "ALERT"

            if force_drowsy: current_eye_state = "DROWSY"

            # C. SIMULATION
            if time.time() - last_sim_update > 1.5:
                step += 1
                if step < 10: sim_speed, sim_loc = 40, "Low"
                elif step < 20: sim_speed, sim_loc = 70, "Medium"
                elif step < 35: sim_speed, sim_loc = 110, "High"
                else: sim_speed, sim_loc = 30, "Low"
                last_sim_update = time.time()

            # D. FUSION ENGINE
            risk_label, risk_val, risk_color = calculate_fusion_risk(sim_speed, 50, sim_loc, current_eye_state, user_history)

            # E. UI UPDATES
            speed_box.markdown(f"<div class='kpi-value'>{sim_speed} km/h</div>", unsafe_allow_html=True)
            hist_box.markdown(f"<div class='kpi-value warn'>{user_history['risk_profile']}</div>", unsafe_allow_html=True)
            
            e_col = "bad" if current_eye_state == "DROWSY" else "good"
            eye_box.markdown(f"<div class='kpi-value {e_col}'>{current_eye_state}</div>", unsafe_allow_html=True)
            risk_box.markdown(f"<div class='kpi-value {risk_color}'>{risk_val}%</div>", unsafe_allow_html=True)
            
            # Draw Status on Camera
            color = (255, 0, 0) if current_eye_state == "DROWSY" else (0, 255, 0)
            cv2.putText(rgb, f"STATUS: {current_eye_state}", (20, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)
            
            cam_place.image(rgb, channels="RGB", use_container_width=True)
            event_log.info(f"Loc: {sim_loc} | Eyes Visible: {eyes_detected}")
            
            st.session_state["trip_data"].append({
                "time": datetime.now().strftime("%H:%M:%S"),
                "speed": sim_speed, "risk": risk_val, "status": risk_label, "eyes": current_eye_state
            })

        cap.release()

# ==========================================
# 6. REPORT VIEW (Mock AI)
# ==========================================
elif st.session_state["view_mode"] == "report":
    st.markdown("<div class='main-title'>Safety Report</div>", unsafe_allow_html=True)
    
    if not st.session_state["trip_data"]:
        st.warning("No data recorded.")
    else:
        if st.session_state["ai_report"] == "":
            with st.spinner("Generating Professional Safety Analysis..."):
                time.sleep(2) # Fake delay for realism
                data = st.session_state["trip_data"]
                
                # Calculate real stats
                avg_risk = int(sum(d['risk'] for d in data) / len(data))
                max_speed = max(d['speed'] for d in data)
                drowsy_count = sum(1 for d in data if d['eyes'] == "DROWSY")
                
                # Get the SMART Mock Response
                st.session_state["ai_report"] = get_mock_ai_response(avg_risk, max_speed, drowsy_count)
        
        st.markdown(f"""
        <div style="background-color:#161b22; padding:25px; border-radius:15px; border:1px solid #30363d; color:#E6EDF3; margin-bottom: 20px;">
            {st.session_state["ai_report"]}
        </div>
        """, unsafe_allow_html=True)
        
        report_text = f"SAFETY REPORT:\n{st.session_state['ai_report']}\n\nLOGS:\n{pd.DataFrame(st.session_state['trip_data']).to_string()}"
        
        st.download_button("📥 Download Report", report_text, "report.txt")
        if st.button("Back"):
            st.session_state["view_mode"] = "dashboard"
            st.rerun()
#intelligent analysis