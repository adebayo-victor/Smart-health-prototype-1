import os
import asyncio
from flask import Flask, render_template, request, jsonify, Response
from flask_sqlalchemy import SQLAlchemy
from datetime import datetime
from dotenv import load_dotenv
from groq import Groq
from openai import OpenAI  # We use OpenAI SDK to talk to LiteLLM
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
groq_client = Groq(api_key=os.getenv("GROQ_API_KEY"))

litellm_client = OpenAI(
    api_key=os.getenv("LITELLM_API_KEY"),
    base_url=os.getenv("LITELLM_BASE_URL")
)

# ==========================================
# EDGE-TTS SERVICE LAYER (Inspired by backend/services/edge_tts_service.py)
# ==========================================
VOICE_MAP = {
    'ha': {'female': 'ha-NG-MaryamNeural', 'male': 'ha-NG-AliNeural'},
    'pcm': {'female': 'en-NG-EzinneNeural', 'male': 'en-NG-AbeoNeural'},
    'en': {'female': 'en-US-JennyNeural', 'male': 'en-US-GuyNeural'}
}

def resolve_voice(lang: str, gender: str = 'female') -> str:
    """Resolves the appropriate Edge-TTS voice based on language and gender."""
    lang_lower = lang.lower()
    gender_lower = gender.lower()
    
    if lang_lower in VOICE_MAP:
        if gender_lower in VOICE_MAP[lang_lower]:
            return VOICE_MAP[lang_lower][gender_lower]
        return VOICE_MAP[lang_lower]['female']  # Fallback to female
    
    return 'en-US-GuyNeural'  # Default fallback

async def generate_speech_async(text: str, voice: str) -> bytes:
    """
    Generates MP3 audio bytes using edge-tts with robust async streaming.
    Avoids disk I/O by keeping chunks in memory (serverless-safe).
    """
    try:
        communicate = edge_tts.Communicate(
            text=text,
            voice=voice,
            rate="+0%",    # e.g., "-10%" for slower, "+15%" for faster
            volume="+0%",  # e.g., "+10%" or "-10%"
            pitch="+0Hz"   # e.g., "+5Hz" or "-5Hz"
        )
        
        audio_chunks = []
        async for chunk in communicate.stream():
            if chunk["type"] == "audio":
                audio_chunks.append(chunk["data"])
        
        if not audio_chunks:
            raise ValueError("No audio chunks received from Edge-TTS. Voice may be unsupported or IP blocked.")
            
        return b"".join(audio_chunks)
        
    except Exception as e:
        raise RuntimeError(f"Edge-TTS generation failed: {str(e)}")

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
@app.route('/api/tts_instruction', methods=['POST'])
def tts_instruction():
    try:
        data = request.get_json()
        text = data.get('text')
        lang = data.get('language', 'en')
        gender = data.get('gender', 'female')  # New parameter supported by voice resolver

        if not text:
            return jsonify({"error": "No text provided"}), 400

        # 1. Translate text using LiteLLM (if not English)
        lang_map = {'ha': 'Hausa', 'pcm': 'Nigerian Pidgin', 'en': 'English'}
        target_lang_name = lang_map.get(lang, 'English')
        translated_text = text
        
        if lang != 'en':
            translation_response = litellm_client.chat.completions.create(
                model=os.getenv("LLM_MODEL", "gemini/gemini-1.5-flash"),
                messages=[
                    {"role": "system", "content": f"Translate the following text to {target_lang_name}. Return ONLY the translated text, no quotes, no explanation."},
                    {"role": "user", "content": text}
                ],
                temperature=0.3
            )
            translated_text = translation_response.choices[0].message.content.strip()

        # 2. Resolve Voice
        voice = resolve_voice(lang, gender)

        # 3. Generate Audio (Memory-safe, no disk writes)
        audio_bytes = asyncio.run(generate_speech_async(translated_text, voice))
        return Response(audio_bytes, mimetype='audio/mpeg')

    except Exception as e:
        app.logger.error(f"TTS Instruction Error: {str(e)}")
        return jsonify({"error": f"TTS generation failed: {str(e)}"}), 500


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
        app.logger.error(f"Audio Translation Error: {str(e)}")
        return jsonify({"error": str(e)}), 500


if __name__ == '__main__':
    app.run(debug=True, port=5000)
