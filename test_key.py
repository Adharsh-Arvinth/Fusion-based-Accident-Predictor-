import google.generativeai as genai

# PASTE YOUR KEY HERE
KEY = "PASTE_YOUR_AIza_KEY_HERE"

print(f"Testing Key: {KEY[:10]}...") # Prints first 10 chars to check

try:
    genai.configure(api_key=KEY)
    model = genai.GenerativeModel('gemini-pro')
    response = model.generate_content("Hello, are you working?")
    print("\nSUCCESS! Connection established.")
    print("AI Replied:", response.text)
except Exception as e:
    print("\nFAILED.")
    print("ERROR DETAILS:", e)