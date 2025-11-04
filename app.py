from flask import Flask, request, jsonify
from flask_cors import CORS
from dotenv import load_dotenv
from gtts import gTTS
import google.generativeai as genai
import base64
import io
import os

# ------------------ Setup ------------------ #
load_dotenv()
genai.configure(api_key=os.getenv("AI_API_KEY"))

app = Flask(__name__)
CORS(app)

# Model configuration
TEXT_MODEL = "models/gemini-2.5-pro"   # Best reasoning
TTS_ENGINE = "gTTS (Google Text-to-Speech)"  # Free voice engine

# Helper: Get parameter from GET/POST
def get_param(name, default=None):
    if request.method == "GET":
        return request.args.get(name, default)
    elif request.is_json:
        return request.json.get(name, default)
    else:
        return request.form.get(name, default)


# ------------------ TEXT ENDPOINT ------------------ #
@app.route("/api/text", methods=["GET", "POST"])
def handle_text():
    try:
        prompt = get_param("prompt")
        if not prompt:
            return jsonify({"error": "Missing prompt"}), 400

        model = genai.GenerativeModel(TEXT_MODEL)
        response = model.generate_content(prompt)

        return jsonify({
            "response": response.text.strip(),
            "model": TEXT_MODEL
        })
    except Exception as e:
        return jsonify({"error": str(e)}), 500


# ------------------ VOICE ENDPOINT ------------------ #
@app.route("/api/voice", methods=["GET", "POST"])
def handle_voice():
    try:
        prompt = get_param("prompt")
        if not prompt:
            return jsonify({"error": "Missing prompt"}), 400

        # Step 1: Generate AI text response using Gemini 2.5 Pro
        text_model = genai.GenerativeModel(TEXT_MODEL)
        text_resp = text_model.generate_content(prompt)
        text = text_resp.text.strip()

        # Step 2: Convert the response text to speech using gTTS
        tts = gTTS(text=text, lang="en")
        buf = io.BytesIO()
        tts.write_to_fp(buf)
        buf.seek(0)

        # Step 3: Encode MP3 bytes to base64 for JSON transfer
        audio_base64 = base64.b64encode(buf.read()).decode("utf-8")

        return jsonify({
            "text": text,
            "audio_base64": audio_base64,
            "mime_type": "audio/mpeg",
            "text_model": TEXT_MODEL,
            "tts_engine": TTS_ENGINE
        })

    except Exception as e:
        return jsonify({"error": str(e)}), 500


# ------------------ SPEECH TO TEXT ------------------ #
@app.route("/api/voice/transcribe", methods=["POST"])
def transcribe_voice():
    try:
        if "file" not in request.files:
            return jsonify({"error": "Missing audio file"}), 400

        audio_file = request.files["file"]
        audio_bytes = audio_file.read()

        # Transcription using Gemini 2.5 Pro (can handle audio input)
        model = genai.GenerativeModel(TEXT_MODEL)
        response = model.generate_content([audio_bytes])

        return jsonify({"transcription": response.text.strip()})
    except Exception as e:
        return jsonify({"error": str(e)}), 500


# ------------------ ROOT ------------------ #
@app.route("/", methods=["GET"])
def home():
    return jsonify({
        "message": "AI API Layer is running 🚀",
        "models": {
            "text": TEXT_MODEL,
            "voice": TTS_ENGINE
        },
        "endpoints": {
            "GET/POST /api/text": "Text generation (Gemini 2.5 Pro)",
            "GET/POST /api/voice": "Text + Voice output (Pro + gTTS)",
            "POST /api/voice/transcribe": "Speech-to-text"
        }
    })


# ------------------ ENTRY ------------------ #
if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port)
