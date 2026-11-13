import os
import json
import random
import string
import asyncio
import logging
import traceback
import re
import threading
import base64
import io
from gtts import gTTS
from functools import wraps
from datetime import datetime

from flask import Flask, render_template, request, jsonify, Response, session, redirect, url_for
from flask_sqlalchemy import SQLAlchemy
from werkzeug.security import generate_password_hash, check_password_hash
from dotenv import load_dotenv
from groq import Groq
from openai import OpenAI
import qrcode
import edge_tts
from edge_tts.exceptions import NoAudioReceived

# ==========================================
# 1. APP CONFIGURATION
# ==========================================
logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)
load_dotenv()

app = Flask(__name__)
app.secret_key = os.getenv("SECRET_KEY", "super-secret-dev-key-change-in-prod")
app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:////tmp/smarthealth.db'
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False
db = SQLAlchemy(app)

# ==========================================
# 2. DATABASE MODELS (ALL MUST BE HERE)
# ==========================================

class MedicalCenter(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    name = db.Column(db.String(100), nullable=False)
    location = db.Column(db.String(100), nullable=True)
    short_code = db.Column(db.String(10), unique=True, nullable=False)

class Worker(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    username = db.Column(db.String(50), unique=True, nullable=False)
    password_hash = db.Column(db.String(200), nullable=False)
    full_name = db.Column(db.String(100), nullable=False)
    worker_code = db.Column(db.String(20), unique=True, nullable=False)
    role = db.Column(db.String(20), nullable=False, default='CHW')
    facility_id = db.Column(db.Integer, db.ForeignKey('medical_center.id'), nullable=False)
    facility = db.relationship('MedicalCenter', backref='workers')

    def set_password(self, password):
        self.password_hash = generate_password_hash(password)
    def check_password(self, password):
        return check_password_hash(self.password_hash, password)

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
    referral_code = db.Column(db.String(20), nullable=True, unique=True)
    referral_qr = db.Column(db.Text, nullable=True)
    origin_worker_id = db.Column(db.Integer, db.ForeignKey('worker.id'), nullable=True)
    origin_facility_id = db.Column(db.Integer, db.ForeignKey('medical_center.id'), nullable=True)
    destination_facility_id = db.Column(db.Integer, db.ForeignKey('medical_center.id'), nullable=True)
    referral_status = db.Column(db.String(20), default='pending')
    
    # TRACKING
    current_stage = db.Column(db.String(20), default='translation')
    status = db.Column(db.String(20), default='active')
    created_at = db.Column(db.DateTime, default=datetime.utcnow)
    updated_at = db.Column(db.DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)

# --- PATIENT-FACING MODELS ---
class Patient(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    phone_number = db.Column(db.String(20), unique=True, nullable=False, index=True)
    display_name = db.Column(db.String(50), nullable=True)
    preferred_language = db.Column(db.String(10), default='ha')
    created_at = db.Column(db.DateTime, default=datetime.utcnow)

class Keyword(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    word = db.Column(db.String(50), unique=True, nullable=False, index=True)

class ChatMessage(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    patient_id = db.Column(db.Integer, db.ForeignKey('patient.id'), nullable=False, index=True)
    role = db.Column(db.String(10), nullable=False)
    original_text = db.Column(db.Text, nullable=False)
    translated_text = db.Column(db.Text, nullable=True)
    created_at = db.Column(db.DateTime, default=datetime.utcnow)

# Association table for Many-to-Many relationship
memory_keywords = db.Table('memory_keywords',
    db.Column('memory_id', db.Integer, db.ForeignKey('memory_index.id'), primary_key=True),
    db.Column('keyword_id', db.Integer, db.ForeignKey('keyword.id'), primary_key=True)
)

class MemoryIndex(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    patient_id = db.Column(db.Integer, db.ForeignKey('patient.id'), nullable=False, index=True)
    context_chunk = db.Column(db.Text, nullable=False)
    created_at = db.Column(db.DateTime, default=datetime.utcnow)
    keywords = db.relationship('Keyword', secondary=memory_keywords, backref='memories')

# ==========================================
# 3. DATABASE INITIALIZATION & SEEDING
# ==========================================
with app.app_context():
    db.create_all()
    
    if MedicalCenter.query.count() == 0:
        mu = MedicalCenter(name="Murtala Muhammed Specialist Hospital", location="Kano", short_code="MU")
        ak = MedicalCenter(name="Aminu Kano Teaching Hospital", location="Kano", short_code="AK")
        db.session.add_all([mu, ak])
        db.session.commit()
        
        w1 = Worker(username="chw_amina", full_name="Amina Bello", worker_code="MU2789", role="CHW", facility_id=mu.id)
        w1.set_password("password123")
        w2 = Worker(username="dr_yusuf", full_name="Dr. Yusuf Ali", worker_code="AK1001", role="Doctor", facility_id=ak.id)
        w2.set_password("password123")
        db.session.add_all([w1, w2])
        db.session.commit()
        logger.info("✅ Seeded facilities and workers.")

# ==========================================
# 4. AI CLIENTS & HELPERS
# ==========================================
groq_client = Groq(api_key=os.getenv("GROQ_API_KEY"))
litellm_client = OpenAI(api_key=os.getenv("LITELLM_API_KEY"), base_url=os.getenv("LITELLM_BASE_URL"))

def login_required(f):
    @wraps(f)
    def decorated_function(*args, **kwargs):
        if 'worker_id' not in session:
            return redirect(url_for('login'))
        return f(*args, **kwargs)
    return decorated_function

def generate_referral_code(dest_short_code):
    while True:
        rand = ''.join(random.choices(string.ascii_uppercase + string.digits, k=6))
        code = f"{dest_short_code}-{rand}"
        if not Encounter.query.filter_by(referral_code=code).first():
            return code

def make_qr_data_uri(text):
    img = qrcode.make(text)
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return "data:image/png;base64," + base64.b64encode(buf.getvalue()).decode("utf-8")

def get_or_create_keyword(word):
    word = word.lower().strip()
    kw = Keyword.query.filter_by(word=word).first()
    if not kw:
        kw = Keyword(word=word)
        db.session.add(kw)
        db.session.flush()
    return kw

def extract_search_keywords(text):
    text = text.lower()
    text = re.sub(r'[^\w\s]', '', text)
    words = text.split()
    stop_words = {'the', 'and', 'for', 'are', 'but', 'not', 'you', 'all', 'can', 'had', 'her', 'was', 'one', 'our', 'out', 'day', 'get', 'has', 'him', 'his', 'how', 'its', 'may', 'new', 'now', 'old', 'see', 'two', 'who', 'boy', 'did', 'she', 'use', 'way', 'many', 'would', 'there', 'their', 'na', 'wetin', 'dey', 'abi', 'sha', 'ko', 'ni', 'ta', 'da', 'wu'}
    keywords = []
    for word in words:
        if word.isdigit() or (len(word) > 3 and word not in stop_words):
            keywords.append(word)
    return list(set(keywords))

def extract_keywords_and_answer(translated_text, past_memories_text):
    prompt = f"""
    You are a trusted, empathetic health worker speaking to a Nigerian woman. 
    Answer her question directly, simply, and calmly. If there are red-flag symptoms, advise her to see a doctor immediately.

    PAST CONTEXT (Her history):
    {past_memories_text if past_memories_text else 'No past history.'}

    CURRENT QUESTION (Translated to English): 
    "{translated_text}"

    Return ONLY valid JSON in this exact format:
    {{
      "answer_english": "The medical answer in simple English.",
      "answer_hausa": "The translation of your medical advice in simple Hausa.",
      "keywords": ["word1", "word2", "number3"] 
    }}
    """
    try:
        response = litellm_client.chat.completions.create(
            model=os.getenv("LLM_MODEL", "gemini/gemini-1.5-flash"),
            messages=[{"role": "user", "content": prompt}],
            temperature=0.3
        )
        raw = response.choices[0].message.content.strip()
        if raw.startswith("```json"): raw = raw[7:]
        if raw.endswith("```"): raw = raw[:-3]
        return json.loads(raw.strip())
    except Exception as e:
        logger.error(f"❌ AI Extraction Error: {e}")
        return {"answer_english": "Error", "answer_hausa": "Na sami matsala, gwada sake.", "keywords": []}

# ==========================================
# 5. FRONTEND ROUTES
# ==========================================
@app.route('/')
def home(): return render_template('start.html')

@app.route('/login', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        username = request.form.get('username')
        password = request.form.get('password')
        worker = Worker.query.filter(
            (Worker.username == username) | (Worker.worker_code == username)
        ).first()
        
        if worker and worker.check_password(password):
            session['worker_id'] = worker.id
            session['facility_id'] = worker.facility_id
            session['worker_code'] = worker.worker_code
            return redirect('/dashboard')
        return render_template('login.html', error="Invalid worker code or password.")
    return render_template('login.html')

@app.route('/logout')
def logout():
    session.clear()
    return redirect('/login')

@app.route('/translator')
@login_required
def translator(): return render_template('translator.html')

@app.route('/triage')
@login_required
def triage(): return render_template('triage.html')

@app.route('/diagnosis')
@login_required
def diagnosis(): return render_template('diagnosis.html')

@app.route('/referral')
@login_required
def referral(): return render_template('referral.html')

@app.route('/lookup')
@login_required
def lookup_page(): return render_template('lookup.html')

@app.route('/patient')
def patient_portal():
    return render_template('patient.html')

@app.route('/dashboard')
@login_required
def dashboard():
    worker_id = session['worker_id']
    facility_id = session['facility_id']
    
    query = Encounter.query.filter(
        db.or_(
            Encounter.origin_worker_id == worker_id,
            Encounter.destination_facility_id == facility_id
        )
    ).order_by(Encounter.created_at.desc())
    
    encounters = query.all()
    facility_centers = {c.id: c.name for c in MedicalCenter.query.all()}
    
    return render_template('dashboard.html', 
                           encounters=encounters, 
                           current_worker_id=worker_id,
                           current_facility_id=facility_id,
                           facility_centers=facility_centers)

# ==========================================
# 6. BACKEND API ROUTES
# ==========================================

@app.route('/api/centers', methods=['GET'])
@login_required
def get_centers():
    centers = MedicalCenter.query.all()
    return jsonify([{"id": c.id, "name": c.name, "location": c.location, "short_code": c.short_code} for c in centers])

@app.route('/api/referral', methods=['POST'])
@login_required
def generate_referral():
    try:
        data = request.get_json()
        patient_name = data.get('patient_name')
        center_id = data.get('center_id')
        reason = data.get('reason')
        encounter_id = data.get('encounter_id')

        if not patient_name or not center_id or not reason:
            return jsonify({"error": "Patient name, center, and reason are required."}), 400

        dest_center = MedicalCenter.query.get(center_id)
        if not dest_center: return jsonify({"error": "Invalid medical center."}), 404

        if encounter_id:
            encounter = Encounter.query.get(encounter_id)
        else:
            encounter = Encounter(patient_name=patient_name)
            db.session.add(encounter)

        code = generate_referral_code(dest_center.short_code)
        qr_uri = make_qr_data_uri(code)

        encounter.referral_reason = reason
        encounter.referral_facility = dest_center.name
        encounter.referral_code = code
        encounter.referral_qr = qr_uri
        encounter.origin_worker_id = session['worker_id']
        encounter.origin_facility_id = session['facility_id']
        encounter.destination_facility_id = dest_center.id
        encounter.referral_status = 'pending'
        encounter.current_stage = 'referral'
        encounter.status = 'referred'
        db.session.commit()

        return jsonify({
            "encounter_id": encounter.id, "referral_code": code, "referral_qr": qr_uri,
            "facility": dest_center.name, "patient_name": patient_name, "reason": reason
        })
    except Exception as e:
        logger.error(f"❌ Referral Error: {traceback.format_exc()}")
        return jsonify({"error": str(e)}), 500

@app.route('/api/lookup', methods=['POST'])
@login_required
def lookup_patient():
    try:
        data = request.get_json()
        code = (data.get('code') or '').strip().upper()
        if not code: return jsonify({"error": "Please enter a referral code."}), 400

        encounter = Encounter.query.filter_by(referral_code=code).first()
        if not encounter: return jsonify({"error": "No patient found with this code."}), 404

        if encounter.destination_facility_id != session['facility_id']:
            return jsonify({"error": "Access Denied: This referral is not addressed to your facility."}), 403

        if encounter.referral_status == 'pending':
            encounter.referral_status = 'fulfilled'
            db.session.commit()

        vitals = json.loads(encounter.vitals_json) if encounter.vitals_json else {}
        return jsonify({
            "patient_name": encounter.patient_name,
            "referral_code": encounter.referral_code,
            "facility": encounter.referral_facility,
            "reason": encounter.referral_reason,
            "referral_status": encounter.referral_status,
            "triage_tier": encounter.triage_tier,
            "triage_notes": encounter.triage_notes,
            "diagnosis": encounter.diagnosis_notes,
            "medications": encounter.diagnosis_medications,
            "vitals": vitals,
            "created_at": encounter.created_at.strftime('%Y-%m-%d %H:%M')
        })
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route('/api/translate', methods=['POST'])
@login_required
def translate_audio():
    try:
        audio_file = request.files.get('audio')
        if not audio_file:
            return jsonify({"error": "No audio file provided."}), 400

        audio_data = audio_file.read()
        if len(audio_data) < 2000:
            return jsonify({"error": "Audio too short. Please record at least 2 seconds."}), 400

        # 1. Transcribe with Groq Whisper
        transcription = groq_client.audio.transcriptions.create(
            file=(audio_file.filename or "audio.webm", audio_data, "audio/webm"),
            model="whisper-large-v3",
            response_format="text"
        )
        raw_text = transcription.strip() if isinstance(transcription, str) else transcription.text.strip()
        if not raw_text:
            return jsonify({"error": "No speech detected. Please try again."}), 400

        # 2. Extract structured medical data with LiteLLM
        prompt = f"""
        You are a medical data extraction assistant. Analyze this patient statement recorded by a health worker in Nigeria.
        The statement may be in Hausa, Nigerian Pidgin, or English.

        STATEMENT: "{raw_text}"

        Return ONLY valid JSON (no markdown, no extra text):
        {{
          "patient_name": "Patient's name if mentioned, otherwise 'Unnamed Patient'",
          "dialect": "Detected language: Hausa, Pidgin, or English",
          "statement": "The original statement exactly as transcribed",
          "translation": "The statement translated into clear English",
          "vitals": {{
            "temp": "temperature in Celsius or 'N/A'",
            "bp_sys": "systolic BP or 'N/A'",
            "bp_dia": "diastolic BP or 'N/A'",
            "spo2": "oxygen saturation % or 'N/A'"
          }},
          "conclusion": "One-sentence clinical summary of the main complaint"
        }}
        """
        response = litellm_client.chat.completions.create(
            model=os.getenv("LLM_MODEL", "gemini/gemini-1.5-flash"),
            messages=[{"role": "user", "content": prompt}],
            temperature=0.2
        )
        raw = response.choices[0].message.content.strip()
        if raw.startswith("```json"): raw = raw[7:]
        elif raw.startswith("```"): raw = raw[3:]
        if raw.endswith("```"): raw = raw[:-3]
        data = json.loads(raw.strip())

        # 3. Persist to the database so it shows on the Dashboard
        encounter = Encounter(
            patient_name=data.get("patient_name", "Unnamed Patient"),
            dialect=data.get("dialect"),
            statement=data.get("statement"),
            translation=data.get("translation"),
            vitals_json=json.dumps(data.get("vitals", {})),
            current_stage="translation",
            status="active"
        )
        db.session.add(encounter)
        db.session.commit()

        data["encounter_id"] = encounter.id
        return jsonify(data)

    except json.JSONDecodeError:
        logger.error("❌ AI returned malformed JSON")
        return jsonify({"error": "AI returned malformed data. Please try again."}), 500
    except Exception as e:
        logger.error(f"❌ Translate Error: {traceback.format_exc()}")
        return jsonify({"error": str(e)}), 500

@app.route('/api/tts_instruction', methods=['POST'])
def tts_instruction():
    try:
        data = request.get_json()
        text = data.get('text')
        language = data.get('language', 'en')

        if not text:
            return jsonify({"error": "No text provided"}), 400

        # 1. Translate the instruction to the target language (if needed)
        target_text = text
        if language != 'en':
            lang_map = {'ha': 'Hausa', 'pcm': 'Nigerian Pidgin'}
            target_lang_name = lang_map.get(language, 'Hausa')
            
            try:
                tr = litellm_client.chat.completions.create(
                    model=os.getenv("LLM_MODEL", "gemini/gemini-1.5-flash"),
                    messages=[
                        {"role": "system", "content": f"Translate the following medical instruction to {target_lang_name}. Return ONLY the translated text, no quotes."},
                        {"role": "user", "content": text}
                    ],
                    temperature=0.3
                )
                target_text = tr.choices[0].message.content.strip()
            except Exception as e:
                logger.error(f"⚠️ Translation failed, using original text: {e}")

        # 2. Generate Audio using lightweight gTTS
        audio_base64 = None
        try:
            # gTTS language codes: 'ha' for Hausa, 'en' for English (and closest match for Pidgin)
            gtts_lang = 'ha' if language == 'ha' else 'en'
            
            tts = gTTS(text=target_text, lang=gtts_lang, slow=False)
            
            # Write to memory (perfect for Vercel serverless, no disk I/O)
            audio_fp = io.BytesIO()
            tts.write_to_fp(audio_fp)
            audio_fp.seek(0)
            
            # Encode to base64 for the frontend
            audio_bytes = audio_fp.read()
            audio_base64 = "data:audio/mp3;base64," + base64.b64encode(audio_bytes).decode('utf-8')
            logger.info(f"✅ gTTS instruction audio generated successfully for {language}.")
            
        except Exception as e:
            logger.error(f"❌ gTTS failed for instruction: {e}")
            return jsonify({"error": "Failed to generate audio."}), 500

        return jsonify({
            "audio_url": audio_base64,
            "translated_text": target_text
        })

    except Exception as e:
        logger.error(f"❌ TTS Instruction Error: {traceback.format_exc()}")
        return jsonify({"error": str(e)}), 500

@app.route('/api/patient_voice', methods=['POST'])
def patient_voice():
    try:
        phone_number = request.form.get('phone_number')
        language = request.form.get('language', 'ha')
        audio_file = request.files.get('audio')

        if not phone_number or not audio_file:
            return jsonify({"error": "Missing phone number or audio."}), 400

        # 🛡️ SAFETY CHECK: Read the file and check its size
        audio_data = audio_file.read()
        file_size = len(audio_data)
        
        # If the file is less than 2KB, it's basically empty (user tapped too fast or mic failed)
        if file_size < 2000:
            logger.warning(f"⚠️ Audio file too small: {file_size} bytes")
            return jsonify({"error": "Audio too short. Please hold the mic and speak for at least 2 seconds."}), 400

        patient = Patient.query.filter_by(phone_number=phone_number).first()
        if not patient:
            patient = Patient(phone_number=phone_number, preferred_language=language)
            db.session.add(patient)
            db.session.flush()

        # 3. Speech to Text (Groq Whisper) with error handling
        try:
            transcription = groq_client.audio.transcriptions.create(
                file=(audio_file.filename or "recording.webm", audio_data, "audio/webm"),
                model="whisper-large-v3",
                prompt=f"This is a medical question in {language}.",
                response_format="text"
            )
        except Exception as groq_err:
            logger.error(f"❌ Groq Transcription Failed: {groq_err}")
            return jsonify({"error": "Could not understand the audio. Please speak clearly and try again."}), 400

        # Groq returns a plain string when response_format="text"
        user_text = transcription.strip() if isinstance(transcription, str) else transcription.text.strip()
        if not user_text:
            return jsonify({"error": "No speech detected. Please try again."}), 400

        translated_text = user_text
        if language != 'en':
            tr = litellm_client.chat.completions.create(
                model=os.getenv("LLM_MODEL", "gemini/gemini-1.5-flash"),
                messages=[{"role": "system", "content": "Translate to English. Return ONLY the translated text."}, {"role": "user", "content": user_text}],
                temperature=0.3
            )
            translated_text = tr.choices[0].message.content.strip()

        short_term = ChatMessage.query.filter_by(patient_id=patient.id).order_by(ChatMessage.created_at.desc()).limit(10).all()
        short_term.reverse()
        short_term_text = "\n".join([f"{m.role}: {m.translated_text or m.original_text}" for m in short_term])

        search_keywords = extract_search_keywords(translated_text)
        long_term_text = ""
        if search_keywords:
            memories = MemoryIndex.query.filter(
                MemoryIndex.patient_id == patient.id,
                MemoryIndex.keywords.any(Keyword.word.in_(search_keywords))
            ).order_by(MemoryIndex.created_at.desc()).limit(5).all()
            long_term_text = "\n".join([f"Past Context: {m.context_chunk}" for m in memories])

        ai_response = extract_keywords_and_answer(translated_text, long_term_text)
        
        answer_english = ai_response.get("answer_english", "")
        answer_hausa = ai_response.get("answer_hausa", answer_english)
        final_keywords = ai_response.get("keywords", [])

        user_msg = ChatMessage(patient_id=patient.id, role='user', original_text=user_text, translated_text=translated_text)
        ai_msg = ChatMessage(patient_id=patient.id, role='ai', original_text=answer_hausa, translated_text=answer_english)
        db.session.add_all([user_msg, ai_msg])

        if final_keywords:
            memory_chunk = f"User: {translated_text}\nAI: {answer_english}"
            new_memory = MemoryIndex(patient_id=patient.id, context_chunk=memory_chunk)
            for kw in final_keywords:
                keyword_obj = get_or_create_keyword(kw)
                new_memory.keywords.append(keyword_obj)
            db.session.add(new_memory)
        
        db.session.commit()

        voice_map = {'ha': 'ha-NG-MaryamNeural', 'pcm': 'en-NG-EzinneNeural', 'en': 'en-US-JennyNeural'}
        voice = voice_map.get(language, 'ha-NG-MaryamNeural')
        audio_base64 = None

        def run_tts():
            nonlocal audio_base64
            try:
                loop = asyncio.new_event_loop()
                asyncio.set_event_loop(loop)
                async def fetch():
                    c = edge_tts.Communicate(text=answer_hausa, voice=voice)
                    chunks = [ch["data"] async for ch in c.stream() if ch["type"] == "audio"]
                    return b"".join(chunks)
                audio_bytes = loop.run_until_complete(fetch())
                audio_base64 = "data:audio/mpeg;base64," + base64.b64encode(audio_bytes).decode('utf-8')
                loop.close()
            except Exception:
                pass

        thread = threading.Thread(target=run_tts)
        thread.start()
        thread.join(timeout=15)

        return jsonify({
            "user_text": user_text,
            "ai_text_hausa": answer_hausa,
            "audio_url": audio_base64
        })

    except Exception as e:
        logger.error(f"❌ Patient Voice Error: {traceback.format_exc()}")
        return jsonify({"error": str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True, port=5000)
