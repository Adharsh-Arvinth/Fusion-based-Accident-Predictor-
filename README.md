# 🚗 Fusion-Based Accident Predictor for Drivers

A comprehensive multi-modal AI system that predicts accident probability by analyzing driver behavior, vehicle dynamics, audio stress, and GPS context in real-time.
![Patent](https://img.shields.io/badge/Patent-Published-success)
![Python](https://img.shields.io/badge/Python-3.11-blue)
![OpenCV](https://img.shields.io/badge/OpenCV-Computer%20Vision-green)
![MediaPipe](https://img.shields.io/badge/MediaPipe-Face%20Tracking-orange)
![Streamlit](https://img.shields.io/badge/Streamlit-Web%20Dashboard-red)
![Google Gemini](https://img.shields.io/badge/Google-Gemini-blueviolet)
![License](https://img.shields.io/badge/License-Research-lightgrey)

## 🎯 Key Features

### Multi-Modal Fusion Analysis
- **Driver Monitoring**: Eye tracking, drowsiness detection, facial tension analysis
- **Vehicle Dynamics**: Steering jerks, harsh braking, speed fluctuations
- **Audio Stress Detection**: Voice stress patterns, shouting detection, cabin noise analysis
- **GPS Context**: Accident-prone zones, speed limit violations, weather conditions
- **Historical Patterns**: Aggressive driving history, near-miss tracking

### Safety Actions
- ✅ Real-time voice alerts ("High accident risk detected—please slow down")
- ✅ Emergency services auto-calling (108/112)
- ✅ GPS location sharing with emergency contacts
- ✅ Preventive safety actions (speed limiter, auto-braking, seat-belt tightening)
- ✅ Hazard light warnings

### Dashboard & Reports
- ✅ Live multi-metric dashboard with 8+ KPIs
- ✅ GPS tracking with accident-prone zone visualization
- ✅ Real-time risk factor display
- ✅ AI-powered post-trip analysis (Gemini AI)
- ✅ Downloadable trip telemetry data

## 🛠️ Technology Stack

- **Computer Vision**: OpenCV, MediaPipe Face Mesh
- **Machine Learning**: Multi-modal fusion algorithms, risk prediction models
- **GenAI**: Google Gemini for intelligent explanations and coaching
- **Audio Processing**: sounddevice, librosa (MFCC features)
- **Backend**: Python, Streamlit
- **Voice Alerts**: pyttsx3 text-to-speech engine

## 📦 Installation

### 1. Install Python Dependencies
```bash
pip install -r requirements.txt
```

### 2. Install System Dependencies (for voice alerts)

**Windows:**
```bash
# pyttsx3 uses Windows SAPI, no additional installation needed
```

**Linux:**
```bash
sudo apt-get install espeak
```

**macOS:**
```bash
# Uses built-in macOS speech synthesis, no additional installation needed
```

### 3. Configure Gemini API Key
Replace the API key in `app.py` line 23 with your own:
```python
GEMINI_API_KEY = "your-api-key-here"
```

Get your free API key at: https://makersuite.google.com/app/apikey

## 🚀 Usage

### Start the Application
```bash
streamlit run app.py
```

### Dashboard Controls
1. Click **"▶ START TRIP"** to begin monitoring
2. The system will:
   - Activate your webcam for driver monitoring
   - Start real-time fusion analysis
   - Display live metrics and GPS tracking
   - Trigger voice alerts when risks are detected
   - Auto-call emergency services if critical risk detected

3. Click **"⏹ END TRIP & REPORT"** to:
   - Stop monitoring
   - Generate AI-powered safety report
   - View detailed telemetry logs
   - Download trip data

### Testing Features
- **Eye Strictness Slider**: Adjust drowsiness detection sensitivity (5-25)
- **Force Drowsy**: Test drowsiness detection without closing eyes
- **Force Emergency**: Test critical alert and emergency calling system

## 📊 System Inputs

### 1. Driver Camera Inputs
- Eye movement tracking
- Facial tension detection
- Yawning detection
- Head nodding
- Emotional stress analysis

### 2. Vehicle Dynamic Inputs
- Steering wheel jerks
- Sudden braking force
- Rapid acceleration
- Lane drift behavior
- Speed fluctuations

### 3. Audio & Voice Inputs
- Driver shouting detection
- Voice stress patterns
- Cabin noise spikes
- Horn sound detection

### 4. GPS & Location Context
- High-accident zones database
- Sharp curves detection
- Speed-limit violation zones
- Weather-based road risk
- Highway vs. city road classification

### 5. Historical Driving Patterns
- Past harsh braking events
- Aggressive driving frequency
- Near-miss detection history

## 🎯 Risk Scoring Algorithm

The system uses weighted multi-modal fusion:
- **Driver State**: 40% weight (drowsiness, facial tension)
- **Vehicle Dynamics**: 30% weight (steering, braking, speed)
- **Audio Stress**: 15% weight (voice stress, shouting)
- **GPS Context**: 15% weight (zone risk, speeding, weather)

### Risk Levels
- **LOW (0-25%)**: Normal driving, all systems green
- **MEDIUM (25-50%)**: Minor concerns, monitoring active
- **HIGH (50-75%)**: Multiple risk factors, voice warnings issued
- **CRITICAL (75-100%)**: Imminent accident risk, emergency protocol activated

## 🚨 Emergency Protocol

When CRITICAL risk is detected (>75%):
1. ⚠️ Visual emergency banner displayed
2. 🔊 Voice alert: "Critical Warning. Accident Imminent..."
3. 📞 Auto-call emergency services (108/112)
4. 📍 GPS location shared with emergency contacts
5. 🚗 Preventive actions: Auto-brakes, speed limiter, seat-belt tightening
6. 🚨 Hazard lights activated

## 📈 Future Enhancements

- [ ] Real CAN bus integration for actual vehicle data
- [ ] Live microphone audio stress analysis
- [ ] MediaPipe Face Mesh for advanced facial analysis
- [ ] LSTM/GRU models for driving sequence prediction
- [ ] Mobile app integration (Android Auto/CarPlay)
- [ ] Cloud-based historical pattern analysis
- [ ] Multi-vehicle fleet management dashboard

## 🔒 Privacy & Security

- All processing happens locally on-device
- No video/audio data is transmitted or stored externally
- GPS data is only shared during emergency events
- Trip logs can be deleted after each session

## 📝 License

This project is for educational and research purposes. For commercial deployment in vehicles, ensure compliance with automotive safety standards and regulations.
## 📜 Patent & Intellectual Property

This project and its underlying multi-modal data fusion architecture are legally protected under a published patent. 

* **Patent Title:** Multi-Modal Accident Prediction and Real-Time Driver Intervention System
* **Status:** Published

### 📝 Usage & Licensing
This repository is provided strictly for **educational, research, and evaluation purposes**. 
* Commercial deployment, duplication of the underlying predictive algorithms, or integration into production vehicular systems without explicit written permission from the patent owners is strictly prohibited.
* For licensing inquiries or commercial use, please contact the author via the [Support](#-support) section.


## 🤝 Contributing

Contributions are welcome! Areas for improvement:
- Real-time audio processing implementation
- Advanced ML models for risk prediction
- Integration with actual vehicle ECU systems
- Mobile app development

## 📧 Support

For issues or questions, please open an issue on the repository.

---

**⚠️ Disclaimer**: This is a prototype system for demonstration purposes. For actual vehicle deployment, professional automotive engineering and safety certification is required.
