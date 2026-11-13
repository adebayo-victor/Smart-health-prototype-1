import os
import asyncio
import logging
import traceback
import json
from flask import Flask, render_template, request, jsonify, Response
from flask_sqlalchemy import SQLAlchemy
from datetime import datetime
from dotenv import load_dotenv
from groq import Groq
from openai import OpenAI
import edge_tts
from edge_tts.exceptions import NoAudioReceived

# Configure logging
logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

load_dotenv()

app = Flask(__name__)

# ==========================================
# 1. NEW MODULAR DATABASE CONFIGURATION
# ==========================================
app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:////tmp/smarthealth.db'
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False
db = SQLAlchemy(app)

class Encounter(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    patient_name = db.Column(db.String(100), nullable=False)
    
    # STAGE 1: TRANSLATION
    dialect = db.Column(db.String(50), nullable=True)
    statement = db.Column(db.Text, nullable=True)
    translation = db.Column(db.Text, nullable=True)
    vitals_json = db.Column(db.Text, nullable=True)
    tc_steps_translation = db.Column(db.Text, nullable=True)
    
    # STAGE 2: TRIAGE
    triage_tier = db.Column(db.String(10), nullable=True)
    triage_notes = db.Column(db.Text, nullable=True)
    tc_steps_triage = db.Column(db.Text, nullable=True)
    
    # STAGE 3: DIAGNOSIS
    diagnosis_notes = db.Column(db.Text, nullable=True)
    diagnosis_medications = db.Column(db.Text, nullable=True)
    tc_steps_diagnosis = db.Column(db.Text, nullable=True)
    
    # STAGE 4: REFERRAL
    referral_reason = db.Column(db.Text, nullable=True)
    referral_facility = db.Column(db.String(100), nullable=True)
    tc_steps_referral = db.Column(db.Text, nullable=True)
    
    # TRACKING
    current_stage = db.Column(db.String(20), default='translation')
    status = db.Column(db.String(20), default='active')
    created_at = db.Column(db.DateTime, default=datetime.utcnow)
    updated_at = db.Column(db.DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)

with app.app_context():
    db.create_all()

# ==========================================
# 2. AI CLIENTS
# ==========================================
groq_client = Groq(api_key=os.getenv("GROQ_API_KEY"))
litellm_client = OpenAI(
    api_key=os.getenv("LITELLM_API_KEY"),
    base_url=os.getenv("LITELLM_BASE_URL")
)

# ==========================================
# 3. HELPER: Get or Create Encounter
# ==========================================
def get_or_create_encounter(data):
    encounter_id = data.get('encounter_id')
    patient_name = data.get('patient_name', 'Unknown Patient')
    
    if encounter_id:
        encounter = Encounter.query.get(encounter_id)
        if not encounter:
            raise ValueError("Invalid encounter_id")
        if patient_name != 'Unknown Patient' and encounter.patient_name == 'Unknown Patient':
            encounter.patient_name = patient_name
    else:
        encounter = Encounter(patient_name=patient_name)
        db.session.add(encounter)
    return encounter

# ==========================================
# 4. FRONTEND ROUTES
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
# 5. BACKEND API ROUTES
# ==========================================

# --- TRIAGE ROUTE (THE MISSING PIECE) ---
@app.route('/api/triage', methods=['POST'])
def triage_assessment():
    try:
        data = request.get_json()
        encounter = get_or_create_encounter(data)
        
        # Parse vitals (allow None if not provided)
        temp = data.get('temp')
        sys_bp = data.get('sys_bp')
        dia_bp = data.get('dia_bp')
        spo2 = data.get('spo2')
        hr = data.get('hr')
        age = data.get('age')
        gender = data.get('gender', '').lower()
        meds = data.get('meds', '')

        tier = 'GREEN'
        triggers = []
        notes = []

        # 1. Temperature Logic
        if temp is not None:
            if temp >= 39.0 or temp <= 34.9:
                tier = 'RED'
                triggers.append(f"Temperature ({temp}°C)")
                notes.append("Critical fever or hypothermia.")
            elif 37.5 <= temp <= 38.9 and tier != 'RED':
                tier = 'YELLOW'
                triggers.append(f"Temperature ({temp}°C)")
                notes.append("Elevated temperature.")

        # 2. Blood Pressure Logic
        if sys_bp is not None or dia_bp is not None:
            if (sys_bp is not None and sys_bp >= 140) or (dia_bp is not None and dia_bp >= 90):
                tier = 'RED'
                bp_str = f"{sys_bp or '?'}/{dia_bp or '?'}"
                triggers.append(f"Blood Pressure ({bp_str} mmHg)")
                notes.append("Hypertensive crisis / Preeclampsia risk.")
            elif ((sys_bp is not None and 120 <= sys_bp <= 139) or (dia_bp is not None and 80 <= dia_bp <= 89)) and tier != 'RED':
                tier = 'YELLOW'
                bp_str = f"{sys_bp or '?'}/{dia_bp or '?'}"
                triggers.append(f"Blood Pressure ({bp_str} mmHg)")
                notes.append("Elevated blood pressure.")

        # 3. SpO2 Logic
        if spo2 is not None:
            if spo2 < 90:
                tier = 'RED'
                triggers.append(f"SpO2 ({spo2}%)")
                notes.append("Severe hypoxemia.")
            elif 90 <= spo2 <= 94 and tier != 'RED':
                tier = 'YELLOW'
                triggers.append(f"SpO2 ({spo2}%)")
                notes.append("Mild hypoxemia.")

        # 4. Heart Rate Logic
        if hr is not None:
            if hr > 120 or hr < 50:
                tier = 'RED'
                triggers.append(f"Heart Rate ({hr} BPM)")
                notes.append("Severe tachycardia or bradycardia.")
            elif (101 <= hr <= 119 or 50 <= hr <= 59) and tier != 'RED':
                tier = 'YELLOW'
                triggers.append(f"Heart Rate ({hr} BPM)")
                notes.append("Mild tachycardia or bradycardia.")

        # 5. Age/Gender Multiplier (Pregnancy Risk)
        if age is not None and 15 <= int(age) <= 49 and gender == 'female':
            if tier == 'GREEN' and ((sys_bp is not None and sys_bp >= 130) or (dia_bp is not None and dia_bp >= 85)):
                tier = 'YELLOW'
                triggers.append("BP in pregnant woman")
                notes.append("Elevated BP in reproductive-age female requires caution.")

        # Format Output
        trigger_text = ", ".join(triggers) if triggers else "All recorded vitals within stable ranges"
        note_text = " ".join(notes) if notes else "Patient is stable."
        
        is_limited = (temp is None and sys_bp is None and dia_bp is None and spo2 is None and hr is None)
        limited_note = " (Assessment based on limited vitals)" if not is_limited and len(triggers) < 2 else ""

        final_notes = f"{note_text}{limited_note}"

        # TC Steps for Triage
        if tier == 'RED':
            tc_steps = ["Re-check all vitals immediately to confirm.", "Prepare for immediate referral."]
        elif tier == 'YELLOW':
            tc_steps = ["Monitor patient closely.", "Re-assess vitals in 1-2 hours."]
        else:
            tc_steps = ["Routine follow-up as needed."]

        # Save to DB
        encounter.triage_tier = tier
        encounter.triage_notes = final_notes
        encounter.tc_steps_triage = json.dumps(tc_steps)
        
        # Save raw vitals for context
        vitals_dict = {k: v for k, v in {'temp': temp, 'sys_bp': sys_bp, 'dia_bp': dia_bp, 'spo2': spo2, 'hr': hr, 'age': age, 'gender': gender, 'meds': meds}.items() if v is not None}
        encounter.vitals_json = json.dumps(vitals_dict)
        encounter.current_stage = 'triage'
        
        db.session.commit()

        return jsonify({
            "encounter_id": encounter.id,
            "triage_data": {
                "tier": tier,
                "trigger": trigger_text,
                "notes": final_notes,
                "tc_steps": tc_steps
            }
        })
    except Exception as e:
        logger.error(f"❌ Triage Error: {traceback.format_exc()}")
        return jsonify({"error": str(e), "traceback": traceback.format_exc()}), 500


# --- TRANSLATION ROUTE (Updated to require patient_name and use new model) ---
@app.route('/api/translate', methods=['POST'])
def translate_audio():
    try:
        data = request.form
        patient_name = data.get('patient_name')
        if not patient_name:
            return jsonify({"error": "Patient name is required"}), 400
            
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
        
        # For now, keeping your mock result structure but mapping to new DB columns
        mock_result = {
            "dialect": "Hausa",
            "statement": raw_text,
            "translation": "Patient reports fever and headache since yesterday.",
            "vitals": {"temp": 38.5, "bp_sys": 120, "bp_dia": 80, "spo2": 98},
            "conclusion": "Acute febrile illness. Monitor for malaria.",
            "tc_steps": ["Check patient's forehead for heat", "Ask patient to point to the pain area"]
        }
        
        new_encounter = Encounter(
            patient_name=patient_name,
            dialect=mock_result['dialect'],
            statement=mock_result['statement'],
            translation=mock_result['translation'],
            vitals_json=json.dumps(mock_result['vitals']),
            tc_steps_translation=json.dumps(mock_result['tc_steps']),
            conclusion=mock_result['conclusion'],
            tier='GREEN'
        )
        db.session.add(new_encounter)
        db.session.commit()
        
        return jsonify({
            "encounter_id": new_encounter.id,
            "patient_name": patient_name,
            "dialect": mock_result['dialect'],
            "statement": mock_result['statement'],
            "translation": mock_result['translation'],
            "vitals": mock_result['vitals'],
            "conclusion": mock_result['conclusion'],
            "tc_steps": mock_result['tc_steps']
        })
    except Exception as e:
        logger.error(f"❌ Audio Translation Error: {traceback.format_exc()}")
        return jsonify({"error": str(e), "traceback": traceback.format_exc()}), 500


# --- TTS ROUTE (Kept exactly as you had it) ---
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

        voice_map = {
            'ha': {'female': 'ha-NG-MaryamNeural', 'male': 'ha-NG-AliNeural'},
            'pcm': {'female': 'en-NG-EzinneNeural', 'male': 'en-NG-AbeoNeural'},
            'en': {'female': 'en-US-JennyNeural', 'male': 'en-US-GuyNeural'}
        }
        voice = voice_map.get(lang, {}).get(gender, 'en-US-GuyNeural')
        logger.debug(f"🎙️ Using Edge-TTS Voice: {voice}")

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


if __name__ == '__main__':
    app.run(debug=True, port=5000)
