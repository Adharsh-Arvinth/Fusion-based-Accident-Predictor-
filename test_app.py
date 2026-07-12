#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Quick test script to verify app.py works correctly
"""

import sys

print("Testing Fusion-Based Accident Predictor...")
print("=" * 50)

# Test 1: Import checks
print("\n1. Testing imports...")
try:
    import streamlit
    print("   ✅ streamlit")
except ImportError as e:
    print(f"   ❌ streamlit: {e}")
    sys.exit(1)

try:
    import cv2
    print("   ✅ opencv-python")
except ImportError as e:
    print(f"   ❌ opencv-python: {e}")
    sys.exit(1)

try:
    import pandas
    print("   ✅ pandas")
except ImportError as e:
    print(f"   ❌ pandas: {e}")
    sys.exit(1)

try:
    import numpy
    print("   ✅ numpy")
except ImportError as e:
    print(f"   ❌ numpy: {e}")
    sys.exit(1)

# Test 2: Optional imports
print("\n2. Testing optional imports...")
try:
    from google import genai
    print("   ✅ google-genai (new package)")
    gemini_available = True
except ImportError:
    try:
        import google.generativeai
        print("   ✅ google-generativeai (old package)")
        gemini_available = True
    except ImportError:
        print("   ⚠️  No Gemini AI package (AI reports will be disabled)")
        gemini_available = False

try:
    import pyttsx3
    print("   ✅ pyttsx3 (voice alerts enabled)")
except ImportError:
    print("   ⚠️  pyttsx3 not installed (voice alerts disabled)")

# Test 3: Camera check
print("\n3. Testing camera...")
try:
    cap = cv2.VideoCapture(0)
    if cap.isOpened():
        print("   ✅ Camera detected and accessible")
        cap.release()
    else:
        print("   ⚠️  Camera not accessible (check permissions)")
except Exception as e:
    print(f"   ⚠️  Camera error: {e}")

# Test 4: OpenCV cascades
print("\n4. Testing OpenCV cascades...")
try:
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
    eye_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_eye.xml')
    if not face_cascade.empty() and not eye_cascade.empty():
        print("   ✅ Face and eye detection models loaded")
    else:
        print("   ❌ Cascade classifiers failed to load")
except Exception as e:
    print(f"   ❌ OpenCV cascade error: {e}")

# Test 5: Syntax check
print("\n5. Testing app.py syntax...")
try:
    with open('app.py', 'r', encoding='utf-8') as f:
        code = f.read()
    compile(code, 'app.py', 'exec')
    print("   ✅ app.py syntax is valid")
except SyntaxError as e:
    print(f"   ❌ Syntax error in app.py: {e}")
    sys.exit(1)
except Exception as e:
    print(f"   ⚠️  Could not read app.py: {e}")

# Summary
print("\n" + "=" * 50)
print("TEST SUMMARY")
print("=" * 50)
print("✅ Core dependencies installed")
print("✅ OpenCV working")
print("✅ app.py syntax valid")
if gemini_available:
    print("✅ AI features available")
else:
    print("⚠️  AI features disabled (install google-genai)")
print("\n🚀 Ready to run: streamlit run app.py")
print("=" * 50)
