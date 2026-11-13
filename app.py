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
    
    # Scoped Query: Show referrals I made OR referrals meant for my facility
    query = Encounter.query.filter(
        db.or_(
            Encounter.origin_worker_id == worker_id,
            Encounter.destination_facility_id == facility_id
        )
    ).order_by(Encounter.created_at.desc())
    
    # (Optional: add your existing tier/stage filters here if needed)
    encounters = query.all()
    return render_template('dashboard.html', encounters=encounters)

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
