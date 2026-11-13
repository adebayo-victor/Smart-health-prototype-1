import os
import asyncio
import logging
import traceback
from flask import Flask, render_template, request, jsonify, Response
from flask_sqlalchemy import SQLAlchemy
from datetime import datetime
from dotenv import load_dotenv
from groq import Groq
from openai import OpenAI
import edge_tts
from edge_tts.exceptions import NoAudioReceived

# Configure logging to output to Vercel's console
logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

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
# BACKEND API ROUTES (DIAGNOSTIC MODE)
# ==========================================
@app.route('/api/tts_instruction', methods=['POST'])
def tts_instruction():
    try:
        data = request.get_json()
        logger.debug(f"🔍 Received TTS request payload: {data}")
        
        text = data.get('text')
        lang = data.get('language', 'en')
        gender = data.get('gender', 'female')

        if not text:
            return jsonify({"error": "No text provided"}), 400

        # 1. Translate text using LiteLLM (if not English)
        lang_map = {'ha': 'Hausa', 'pcm': 'Nigerian Pidgin', 'en': 'English'}
        target_lang_name = lang_map.get(lang, 'English')
        translated_text = text
        
        if lang != 'en':
            logger.debug(f"🔄 Translating to {target_lang_name}...")
            try:
                translation_response = litellm_client.chat.completions.create(
                    model=os.getenv("LLM_MODEL", "gemini/gemini-1.5-flash"),
                    messages=[
                        {"role": "system", "content": f"Translate to {target_lang_name}. Return ONLY the translated text."},
                        {"role": "user", "content": text}
                    ],
                    temperature=0.3
                )
                translated_text = translation_response.choices[0].message.content.strip()
                logger.debug(f"✅ Translation successful: {translated_text}")
            except Exception as e:
                logger.error(f"❌ LiteLLM Translation failed: {traceback.format_exc()}")
                return jsonify({"error": f"Translation failed: {str(e)}"}), 500

        # 2. Resolve Voice
        voice_map = {
            'ha': {'female': 'ha-NG-MaryamNeural', 'male': 'ha-NG-AliNeural'},
            'pcm': {'female': 'en-NG-EzinneNeural', 'male': 'en-NG-AbeoNeural'},
            'en': {'female': 'en-US-JennyNeural', 'male': 'en-US-GuyNeural'}
        }
        voice = voice_map.get(lang, {}).get(gender, 'en-US-GuyNeural')
        logger.debug(f"🎙️ Using Edge-TTS Voice: {voice}")

        # 3. Generate Audio
        logger.debug("⏳ Starting edge-tts stream generation...")
        try:
            async def generate_speech_async():
                communicate = edge_tts.Communicate(
                    text=translated_text,
                    voice=voice,
                    rate="+0%",
                    volume="+0%",
                    pitch="+0Hz"
                )
                audio_chunks = []
                async for chunk in communicate.stream():
                    if chunk["type"] == "audio":
                        audio_chunks.append(chunk["data"])
                
                if not audio_chunks:
                    raise ValueError("No audio chunks received from Edge-TTS stream.")
                return b"".join(audio_chunks)

            audio_bytes = asyncio.run(generate_speech_async())
            logger.debug(f"🎉 Successfully generated {len(audio_bytes)} bytes of audio.")
            return Response(audio_bytes, mimetype='audio/mpeg')
            
        except NoAudioReceived as e:
            error_trace = traceback.format_exc()
            logger.error(f"🚨 Edge-TTS NoAudioReceived: {error_trace}")
            return jsonify({
                "error": "No audio received from Microsoft Edge-TTS.",
                "detail": "This is almost certainly a datacenter IP block by Microsoft on Vercel.",
                "traceback": error_trace
            }), 500
            
        except Exception as e:
            error_trace = traceback.format_exc()
            logger.error(f"🚨 Edge-TTS generation failed: {error_trace}")
            return jsonify({
                "error": "Edge-TTS generation failed",
                "detail": str(e),
                "traceback": error_trace
            }), 500

    except Exception as e:
        error_trace = traceback.format_exc()
        logger.error(f"💥 Global TTS Error: {error_trace}")
        return jsonify({
            "error": "Internal Server Error",
            "detail": str(e),
            "traceback": error_trace
        }), 500


@app.route('/api/translate', methods=['POST'])
def translate_audio():
    try:
        if 'audio' not in request.files:
            return jsonify({"error": "No audio file provided"}), 400
        
        audio_file = request.files['audio']
        logger.debug(f"🎤 Received audio file: {audio_file.filename}")
        
        transcription = groq_client.audio.transcriptions.create(
            file=(audio_file.filename, audio_file.read(), "audio/webm"),
            model="whisper-large-v3",
            prompt="This is a medical intake in Nigerian Pidgin or Hausa.",
            response_format="json"
        )
        
        raw_text = transcription.text
        logger.debug(f"✅ Groq STT successful: {raw_text}")
        
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
        logger.error(f"❌ Audio Translation Error: {traceback.format_exc()}")
        return jsonify({"error": str(e), "traceback": traceback.format_exc()}), 500


if __name__ == '__main__':
    app.run(debug=True, port=5000)
