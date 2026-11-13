import os
from flask import Flask, render_template, request, jsonify
from dotenv import load_dotenv
from groq import Groq

# Load environment variables
load_dotenv()

app = Flask(__name__)

# Initialize Groq client
groq_client = Groq(api_key=os.getenv("GROQ_API_KEY"))

# ==========================================
# 1. FRONTEND ROUTES (Serving HTML Pages)
# ==========================================

@app.route('/')
def home():
    return render_template('start.html')

@app.route('/login')
def login():
    return render_template('login.html')

@app.route('/translator')
def translator():
    return render_template('translator.html')

# ==========================================
# 2. BACKEND API ROUTES (The AI Logic)
# ==========================================

@app.route('/api/translate', methods=['POST'])
def translate_audio():
    try:
        if 'audio' not in request.files:
            return jsonify({"error": "No audio file provided"}), 400
        
        audio_file = request.files['audio']
        
        # Send to Groq for Speech-to-Text
        transcription = groq_client.audio.transcriptions.create(
            file=(audio_file.filename, audio_file.read(), "audio/webm"),
            model="whisper-large-v3",
            prompt="This is a medical intake in Nigerian Pidgin or Hausa.",
            response_format="json"
        )
        raw_text = transcription.text

        # Mock LiteLLM response for now (We will wire the real one in Phase 2)
        mock_result = {
            "dialect": "Hausa",
            "statement": raw_text,
            "translation": "Patient reports fever and headache since yesterday.",
            "vitals": {"temp": 38.5, "bp_sys": 120, "bp_dia": 80, "spo2": 98},
            "conclusion": "Acute febrile illness. Monitor for malaria.",
            "source": "Nigerian Standard Treatment Guidelines"
        }
        
        return jsonify(mock_result)

    except Exception as e:
        return jsonify({"error": str(e)}), 500

# Add these to your app.py

@app.route('/triage')
def triage():
    return render_template('triage.html')

@app.route('/diagnosis')
def diagnosis():
    return render_template('diagnosis.html')

@app.route('/referral')
def referral():
    return render_template('referral.html')

if __name__ == '__main__':
    app.run(debug=True, port=5000)
