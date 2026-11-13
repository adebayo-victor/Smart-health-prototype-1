import os
from flask import Flask, render_template, request, jsonify
from flask_sqlalchemy import SQLAlchemy
from datetime import datetime
from dotenv import load_dotenv
from groq import Groq

# Load environment variables
load_dotenv()

app = Flask(__name__)

# ==========================================
# DATABASE CONFIGURATION
# ==========================================
# Create a local SQLite database file named 'smarthealth.db'
app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:////tmp/smarthealth.db'
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False
db = SQLAlchemy(app)

# Define the 'Encounter' table structure
class Encounter(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    patient_name = db.Column(db.String(100), nullable=False, default='Unknown Patient')
    dialect = db.Column(db.String(50))
    statement = db.Column(db.Text)
    translation = db.Column(db.Text)
    conclusion = db.Column(db.Text)
    tier = db.Column(db.String(10), default='GREEN')
    created_at = db.Column(db.DateTime, default=datetime.utcnow)

# Create the database tables when the app starts
with app.app_context():
    db.create_all()

# Initialize Groq client
groq_client = Groq(api_key=os.getenv("GROQ_API_KEY"))

# ==========================================
# FRONTEND ROUTES
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

@app.route('/triage')
def triage():
    return render_template('triage.html')

@app.route('/diagnosis')
def diagnosis():
    return render_template('diagnosis.html')

@app.route('/referral')
def referral():
    return render_template('referral.html')

# NEW: Dashboard Route to show saved encounters
@app.route('/dashboard')
def dashboard():
    # Fetch all encounters from the database, newest first
    encounters = Encounter.query.order_by(Encounter.created_at.desc()).all()
    return render_template('dashboard.html', encounters=encounters)

# ==========================================
# BACKEND API ROUTES
# ==========================================

@app.route('/api/translate', methods=['POST'])
def translate_audio():
    try:
        if 'audio' not in request.files:
            return jsonify({"error": "No audio file provided"}), 400
        
        audio_file = request.files['audio']
        
        # 1. Send to Groq for Speech-to-Text
        transcription = groq_client.audio.transcriptions.create(
            file=(audio_file.filename, audio_file.read(), "audio/webm"),
            model="whisper-large-v3",
            prompt="This is a medical intake in Nigerian Pidgin or Hausa.",
            response_format="json"
        )
        raw_text = transcription.text

        # 2. Mock LiteLLM response (We will wire the real one later)
        mock_result = {
            "dialect": "Hausa",
            "statement": raw_text,
            "translation": "Patient reports fever and headache since yesterday.",
            "vitals": {"temp": 38.5, "bp_sys": 120, "bp_dia": 80, "spo2": 98},
            "conclusion": "Acute febrile illness. Monitor for malaria.",
            "source": "Nigerian Standard Treatment Guidelines"
        }
        
        # 3. SAVE TO DATABASE
        new_encounter = Encounter(
            dialect=mock_result['dialect'],
            statement=mock_result['statement'],
            translation=mock_result['translation'],
            conclusion=mock_result['conclusion'],
            tier='GREEN' # We will calculate this dynamically later
        )
        db.session.add(new_encounter)
        db.session.commit()
        
        return jsonify(mock_result)

    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True, port=5000)
