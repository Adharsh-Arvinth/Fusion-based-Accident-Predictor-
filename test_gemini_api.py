#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Test Gemini API connection and find available models
"""

import sys

API_KEY = "AIzaSyADPgT3Qk-OBcQwqIpgc9oQ8XARsy4Rpt8"

print("=" * 60)
print("Testing Gemini API Connection")
print("=" * 60)

# Test 1: Check if package is installed
print("\n1. Checking installed packages...")
try:
    import google.generativeai as genai
    print("   ✅ google-generativeai package installed")
    package_available = True
except ImportError:
    print("   ❌ google-generativeai not installed")
    print("   Run: pip install google-generativeai")
    package_available = False
    sys.exit(1)

# Test 2: Configure API
print("\n2. Configuring API key...")
try:
    genai.configure(api_key=API_KEY)
    print("   ✅ API key configured")
except Exception as e:
    print(f"   ❌ Configuration error: {e}")
    sys.exit(1)

# Test 3: List available models
print("\n3. Listing available models...")
try:
    models = genai.list_models()
    available_models = []
    
    print("\n   Available models that support generateContent:")
    for model in models:
        if 'generateContent' in model.supported_generation_methods:
            available_models.append(model.name)
            print(f"   ✅ {model.name}")
    
    if not available_models:
        print("   ⚠️  No models found that support generateContent")
        
except Exception as e:
    print(f"   ❌ Error listing models: {e}")
    available_models = []

# Test 4: Try to generate content with each model
print("\n4. Testing content generation...")

test_prompt = "Say 'Hello, I am working!' in one sentence."

models_to_test = [
    'models/gemini-2.5-flash',
    'models/gemini-flash-latest',
    'models/gemini-pro-latest',
    'models/gemini-2.0-flash',
    'gemini-2.5-flash',
    'gemini-flash-latest'
]

working_model = None

for model_name in models_to_test:
    try:
        print(f"\n   Testing {model_name}...")
        model = genai.GenerativeModel(model_name)
        response = model.generate_content(test_prompt)
        print(f"   ✅ {model_name} WORKS!")
        print(f"   Response: {response.text[:50]}...")
        working_model = model_name
        break
    except Exception as e:
        print(f"   ❌ {model_name} failed: {str(e)[:60]}...")
        continue

# Summary
print("\n" + "=" * 60)
print("TEST SUMMARY")
print("=" * 60)

if working_model:
    print(f"✅ SUCCESS! Working model: {working_model}")
    print(f"\nUpdate app.py to use: '{working_model}'")
else:
    print("❌ No working models found")
    print("\nPossible issues:")
    print("• API key may be invalid or expired")
    print("• API quota may be exceeded")
    print("• Network connectivity issues")
    print("• API region restrictions")
    print("\nTroubleshooting:")
    print("1. Verify API key at: https://makersuite.google.com/app/apikey")
    print("2. Check API quota and billing")
    print("3. Try generating a new API key")
    print("4. Check internet connection")

print("=" * 60)
