import os
import asyncio
from flask import Flask, render_template, request, jsonify, Response
from flask_sqlalchemy import SQLAlchemy
from datetime import datetime
from dotenv import load_dotenv
from groq import Groq
from openai import OpenAI # We use OpenAI SDK to talk to LiteLLM
import edge_tts

load_dotenv()

app = Flask(__name__)

# ==========================================
# DATABASE CONFIGURATION
# ==========================================
app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:////tmp/smarthealth.db'
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False
db = SQLAlchemy(app)

class Encounter(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    patient_name = db.Column(db.String(100), nullable=False, default='Unknown Patient')
    dialect = db.Column(db.String(50))
    statement = db.Column(db.Text)
    translation = db.Column(db.Text)
    conclusion = db.Column(db.Text)
    tier = db.Column(db.String(10), default='GREEN')
    created_at = db.Column(db.DateTime, default=datetime.utcnow)

with app.app_context():
    db.create_all()

# ==========================================
# AI CLIENTS
# ==========================================
# 1. Groq Client (For Speech-to-Text / Audio)
groq_client = Groq(api_key=os.getenv("GROQ_API_KEY"))

# 2. LiteLLM Client (For Text Translation / LLM)
# We use the OpenAI SDK but point it to your LiteLLM base URL
litellm_client = OpenAI(
    api_key=os.getenv("LITELLM_API_KEY"),
    base_url=os.getenv("LITELLM_BASE_URL")
)

# ==========================================
# FRONTEND ROUTES
# ==========================================
@app.route('/')
def home(): return render_template('start.html')
@app.route('/login')
def login(): return render_template('login.html')
@app.route('/translator')
def translator(): return render_template('translator.html')
@app.route('/triage')
def triage(): return render_template('triage.html')
@app.route('/diagnosis')
def diagnosis(): return render_template('diagnosis.html')
@app.route('/referral')
def referral(): return render_template('referral.html')
@app.route('/dashboard')
def dashboard():
    encounters = Encounter.query.order_by(Encounter.created_at.desc()).all()
    return render_template('dashboard.html', encounters=encounters)

# ==========================================
# BACKEND API ROUTES
# ==========================================

# NEW: Translate Text and Generate Audio (LiteLLM + Edge-TTS)
@app.route('/api/tts_instruction', methods=['POST'])
def tts_instruction():
    try:
        data = request.get_json()
        text = data.get('text')
        lang = data.get('language', 'en')

        if not text:
            return jsonify({"error": "No text provided"}), 400

        # 1. Translate text using LiteLLM (if not English)
        lang_map = {'ha': 'Hausa', 'pcm': 'Nigerian Pidgin', 'en': 'English'}
        target_lang_name = lang_map.get(lang, 'English')
        
        translated_text = text
        if lang != 'en':
            # Use LiteLLM via OpenAI SDK
            translation_response = litellm_client.chat.completions.create(
                model=os.getenv("LLM_MODEL", "gemini/gemini-1.5-flash"), # Uses your LLM_MODEL secret
                messages=[
                    {"role": "system", "content": f"Translate the following text to {target_lang_name}. Return ONLY the translated text, no quotes, no explanation."},
                    {"role": "user", "content": text}
                ],
                temperature=0.3
            )
            translated_text = translation_response.choices[0].message.content.strip()

        # 2. Generate Audio using Edge-TTS
        voice_map = {
            'ha': 'ha-NG-MajeNeural',
            'pcm': 'en-NG-AbeoNeural', 
            'en': 'en-US-GuyNeural'
        }
        voice = voice_map.get(lang, 'en-US-GuyNeural')

        async def generate_audio(text, voice):
            communicate = edge_tts.Communicate(text, voice)
            audio_data = b""
            async for chunk in communicate.stream():
                if chunk["type"] == "audio":
                    audio_data += chunk["data"]
            return audio_data

        audio_bytes = asyncio.run(generate_audio(translated_text, voice))
        return Response(audio_bytes, mimetype='audio/mpeg')

    except Exception as e:
        # This sends the exact error back to the frontend so we can see it
        return jsonify({"error": str(e)}), 500

# Existing: Audio Transcription (Uses Groq directly for STT)
@app.route('/api/translate', methods=['POST'])
def translate_audio():
    try:
        if 'audio' not in request.files:
            return jsonify({"error": "No audio file provided"}), 400
        
        audio_file = request.files['audio']
        
        transcription = groq_client.audio.transcriptions.create(
            file=(audio_file.filename, audio_file.read(), "audio/webm"),
            model="whisper-large-v3",
            prompt="This is a medical intake in Nigerian Pidgin or Hausa.",
            response_format="json"
        )
        raw_text = transcription.text

        mock_result = {
            "dialect": "Hausa",
            "statement": raw_text,
            "translation": "Patient reports fever and headache since yesterday.",
            "vitals": {"temp": 38.5, "bp_sys": 120, "bp_dia": 80, "spo2": 98},
            "conclusion": "Acute febrile illness. Monitor for malaria.",
            "source": "Nigerian Standard Treatment Guidelines"
        }
        
        new_encounter = Encounter(
            dialect=mock_result['dialect'],
            statement=mock_result['statement'],
            translation=mock_result['translation'],
            conclusion=mock_result['conclusion'],
            tier='GREEN'
        )
        db.session.add(new_encounter)
        db.session.commit()
        
        return jsonify(mock_result)

    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True, port=5000)
