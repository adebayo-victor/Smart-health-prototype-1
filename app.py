import os
import json
import random
import string
import asyncio
import logging
import traceback
from functools import wraps
from flask import Flask, render_template, request, jsonify, Response, session, redirect, url_for
from flask_sqlalchemy import SQLAlchemy
from werkzeug.security import generate_password_hash, check_password_hash
from datetime import datetime
from dotenv import load_dotenv
from groq import Groq
from openai import OpenAI
import qrcode
import io
import base64
import edge_tts
from edge_tts.exceptions import NoAudioReceived

logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)
load_dotenv()

app = Flask(__name__)
app.secret_key = os.getenv("SECRET_KEY", "super-secret-dev-key-change-in-prod") # Required for sessions
app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:////tmp/smarthealth.db'
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False
db = SQLAlchemy(app)

# ==========================================
# 1. DATABASE MODELS (Updated for Facility Scoping)
# ==========================================
class MedicalCenter(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    name = db.Column(db.String(100), nullable=False)
    location = db.Column(db.String(100), nullable=True)
    short_code = db.Column(db.String(10), unique=True, nullable=False) # e.g., 'MU', 'AK'

class Worker(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    username = db.Column(db.String(50), unique=True, nullable=False)
    password_hash = db.Column(db.String(200), nullable=False)
    full_name = db.Column(db.String(100), nullable=False)
    worker_code = db.Column(db.String(20), unique=True, nullable=False) # e.g., 'MU2789'
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
    
    # STAGE 4: REFERRAL (Updated with scoping)
    referral_reason = db.Column(db.Text, nullable=True)
    referral_facility = db.Column(db.String(100), nullable=True)
    referral_code = db.Column(db.String(20), nullable=True, unique=True)
    referral_qr = db.Column(db.Text, nullable=True)
    origin_worker_id = db.Column(db.Integer, db.ForeignKey('worker.id'), nullable=True)
    origin_facility_id = db.Column(db.Integer, db.ForeignKey('medical_center.id'), nullable=True)
    destination_facility_id = db.Column(db.Integer, db.ForeignKey('medical_center.id'), nullable=True)
    referral_status = db.Column(db.String(20), default='pending') # 'pending' or 'fulfilled'
    
    # TRACKING
    current_stage = db.Column(db.String(20), default='translation')
    status = db.Column(db.String(20), default='active')
    created_at = db.Column(db.DateTime, default=datetime.utcnow)
    updated_at = db.Column(db.DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)

with app.app_context():
    db.create_all()
    
    # SEED DATA (Manual fill template)
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
# PATIENT-FACING DATABASE MODELS
# ==========================================

class Patient(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    phone_number = db.Column(db.String(20), unique=True, nullable=False, index=True)
    display_name = db.Column(db.String(50), nullable=True)
    preferred_language = db.Column(db.String(10), default='ha') # 'ha', 'pcm', 'en'
    created_at = db.Column(db.DateTime, default=datetime.utcnow)

class Keyword(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    word = db.Column(db.String(50), unique=True, nullable=False, index=True)

class ChatMessage(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    patient_id = db.Column(db.Integer, db.ForeignKey('patient.id'), nullable=False, index=True)
    
    role = db.Column(db.String(10), nullable=False) # 'user' or 'ai'
    original_text = db.Column(db.Text, nullable=False) # Hausa/Pidgin
    translated_text = db.Column(db.Text, nullable=True) # English
    
    created_at = db.Column(db.DateTime, default=datetime.utcnow)

class MemoryIndex(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    patient_id = db.Column(db.Integer, db.ForeignKey('patient.id'), nullable=False, index=True)
    
    context_chunk = db.Column(db.Text, nullable=False) # The summary of the exchange
    created_at = db.Column(db.DateTime, default=datetime.utcnow)
    
    # The Integer Mapping (Inverted Index)
    keywords = db.relationship('Keyword', secondary='memory_keywords', backref='memories')

# Association table for Many-to-Many relationship between Memory and Keywords
memory_keywords = db.Table('memory_keywords',
    db.Column('memory_id', db.Integer, db.ForeignKey('memory_index.id'), primary_key=True),
    db.Column('keyword_id', db.Integer, db.ForeignKey('keyword.id'), primary_key=True)
)

# ==========================================
# PATIENT-FACING HELPERS
# ==========================================

def get_or_create_keyword(word):
    """Finds a keyword by word, or creates it and returns the ID."""
    word = word.lower().strip()
    kw = Keyword.query.filter_by(word=word).first()
    if not kw:
        kw = Keyword(word=word)
        db.session.add(kw)
        db.session.flush() # Get the ID without committing yet
    return kw

def extract_keywords_and_answer(translated_text, past_memories_text):
    """
    Uses LiteLLM to answer the question AND extract keywords in one go.
    """
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
        # Clean markdown if AI adds it
        if raw.startswith("```json"): raw = raw[7:]
        if raw.endswith("```"): raw = raw[:-3]
        return json.loads(raw.strip())
    except Exception as e:
        logger.error(f"❌ AI Extraction Error: {e}")
        return {"answer_english": "Error", "answer_hausa": "Na sami matsala, gwada sake.", "keywords": []}

import re
import threading
import asyncio
import edge_tts
from edge_tts.exceptions import NoAudioReceived

# ==========================================
# PATIENT-FACING HELPERS (Search Logic)
# ==========================================

def extract_search_keywords(text):
    """
    Quick extraction to find words for the database search.
    Keeps words > 3 letters and numbers.
    """
    text = text.lower()
    text = re.sub(r'[^\w\s]', '', text)
    words = text.split()
    stop_words = {'the', 'and', 'for', 'are', 'but', 'not', 'you', 'all', 'can', 'had', 'her', 'was', 'one', 'our', 'out', 'day', 'get', 'has', 'him', 'his', 'how', 'its', 'may', 'new', 'now', 'old', 'see', 'two', 'who', 'boy', 'did', 'she', 'use', 'way', 'many', 'would', 'there', 'their', 'na', 'wetin', 'dey', 'abi', 'sha', 'ko', 'ni', 'ta', 'da', 'wu'}
    
    keywords = []
    for word in words:
        if word.isdigit() or (len(word) > 3 and word not in stop_words):
            keywords.append(word)
    return list(set(keywords))

# ==========================================
# PATIENT-FACING API ROUTE
# ==========================================

@app.route('/api/patient_voice', methods=['POST'])
def patient_voice():
    try:
        # 1. Get Inputs
        phone_number = request.form.get('phone_number')
        language = request.form.get('language', 'ha')
        audio_file = request.files.get('audio')

        if not phone_number or not audio_file:
            return jsonify({"error": "Missing phone number or audio."}), 400

        # 2. Get or Create Patient
        patient = Patient.query.filter_by(phone_number=phone_number).first()
        if not patient:
            patient = Patient(phone_number=phone_number, preferred_language=language)
            db.session.add(patient)
            db.session.flush()

        # 3. Speech to Text (Groq Whisper)
        transcription = groq_client.audio.transcriptions.create(
            file=(audio_file.filename, audio_file.read(), "audio/webm"),
            model="whisper-large-v3",
            prompt=f"This is a medical question in {language}.",
            response_format="text"
        )
        user_text = transcription.text.strip()
        if not user_text:
            return jsonify({"error": "Could not understand audio."}), 400

        # 4. Translate to English (if not English)
        translated_text = user_text
        if language != 'en':
            lang_map = {'ha': 'Hausa', 'pcm': 'Nigerian Pidgin'}
            tr = litellm_client.chat.completions.create(
                model=os.getenv("LLM_MODEL", "gemini/gemini-1.5-flash"),
                messages=[{"role": "system", "content": f"Translate to English. Return ONLY the translated text."}, {"role": "user", "content": user_text}],
                temperature=0.3
            )
            translated_text = tr.choices[0].message.content.strip()

        # 5. MEMORY RETRIEVAL
        # Short-term: Last 10 messages
        short_term = ChatMessage.query.filter_by(patient_id=patient.id).order_by(ChatMessage.created_at.desc()).limit(10).all()
        short_term.reverse() # Chronological order
        short_term_text = "\n".join([f"{m.role}: {m.translated_text or m.original_text}" for m in short_term])

        # Long-term: Search DB using rough keywords
        search_keywords = extract_search_keywords(translated_text)
        long_term_text = ""
        if search_keywords:
            # Find memories that have ANY of these keywords
            memories = MemoryIndex.query.filter(
                MemoryIndex.patient_id == patient.id,
                MemoryIndex.keywords.any(Keyword.word.in_(search_keywords))
            ).order_by(MemoryIndex.created_at.desc()).limit(5).all()
            
            long_term_text = "\n".join([f"Past Context: {m.context_chunk}" for m in memories])

        # 6. AI CALL (Answer + Final Keywords)
        ai_response = extract_keywords_and_answer(translated_text, long_term_text)
        
        answer_english = ai_response.get("answer_english", "")
        answer_hausa = ai_response.get("answer_hausa", answer_english)
        final_keywords = ai_response.get("keywords", [])

        # 7. SAVE TO DATABASE
        # Save Chat Messages
        user_msg = ChatMessage(patient_id=patient.id, role='user', original_text=user_text, translated_text=translated_text)
        ai_msg = ChatMessage(patient_id=patient.id, role='ai', original_text=answer_hausa, translated_text=answer_english)
        db.session.add_all([user_msg, ai_msg])

        # Save Memory Index with Integer Mapping
        if final_keywords:
            memory_chunk = f"User: {translated_text}\nAI: {answer_english}"
            new_memory = MemoryIndex(patient_id=patient.id, context_chunk=memory_chunk)
            
            for kw in final_keywords:
                keyword_obj = get_or_create_keyword(kw)
                new_memory.keywords.append(keyword_obj)
                
            db.session.add(new_memory)
        
        db.session.commit()

        # 8. TEXT TO SPEECH (Edge-TTS with Threading Fix)
        voice_map = {
            'ha': 'ha-NG-MaryamNeural',
            'pcm': 'en-NG-EzinneNeural',
            'en': 'en-US-JennyNeural'
        }
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

        # 9. Return Response
        return jsonify({
            "user_text": user_text,
            "ai_text_hausa": answer_hausa,
            "audio_url": audio_base64 # Frontend will play this directly
        })

    except Exception as e:
        logger.error(f"❌ Patient Voice Error: {traceback.format_exc()}")
        return jsonify({"error": str(e)}), 500

# ==========================================
# 2. HELPERS & AUTH
# ==========================================
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

groq_client = Groq(api_key=os.getenv("GROQ_API_KEY"))
litellm_client = OpenAI(api_key=os.getenv("LITELLM_API_KEY"), base_url=os.getenv("LITELLM_BASE_URL"))

# ==========================================
# 3. FRONTEND ROUTES
# ==========================================
@app.route('/')
def home(): return render_template('start.html')

@app.route('/login', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        username = request.form.get('username')
        password = request.form.get('password')
        worker = Worker.query.filter_by(username=username).first()
        
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

@app.route('/dashboard')
@login_required
def dashboard():
    worker_id = session['worker_id']
    facility_id = session['facility_id']
    
    # Filter: Show referrals I sent OR referrals meant for my facility
    query = Encounter.query.filter(
        db.or_(
            Encounter.origin_worker_id == worker_id,
            Encounter.destination_facility_id == facility_id
        )
    ).order_by(Encounter.created_at.desc())
    
    # (Optional: Keep your existing tier/stage search filters here if you want)
    encounters = query.all()
    
    # Get a map of all facility IDs to Names for the template
    facility_centers = {c.id: c.name for c in MedicalCenter.query.all()}
    
    return render_template('dashboard.html', 
                           encounters=encounters, 
                           current_worker_id=worker_id,
                           current_facility_id=facility_id,
                           facility_centers=facility_centers)

# ==========================================
# 4. BACKEND API ROUTES
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

        # Generate code WITH destination short code (e.g., "AK-7K92MP")
        code = generate_referral_code(dest_center.short_code)
        qr_uri = make_qr_data_uri(code)

        encounter.referral_reason = reason
        encounter.referral_facility = dest_center.name
        encounter.referral_code = code
        encounter.referral_qr = qr_uri
        
        # SCOPING DATA
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

        # 🔒 SECURITY CHECK: Is this referral meant for MY facility?
        if encounter.destination_facility_id != session['facility_id']:
            return jsonify({"error": "Access Denied: This referral is not addressed to your facility."}), 403

        # ✅ MARK AS FULFILLED
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

# (Keep your existing /api/triage, /api/diagnosis, /api/translate, /api/tts_instruction routes here...)
# Just make sure to add @login_required to them if you want to lock the whole app.

if __name__ == '__main__':
    app.run(debug=True, port=5000)
