from flask import Flask, request, jsonify
from flask_cors import CORS
from dotenv import load_dotenv
import google.generativeai as genai
import base64
import os

# ------------------ Setup ------------------ #
load_dotenv()
genai.configure(api_key=os.getenv("AI_API_KEY"))

app = Flask(__name__)
CORS(app)

# Model configuration
TEXT_MODEL = "models/gemini-2.5-pro"
TTS_MODEL = "models/gemini-1.5-flash-tts"

# Helper: Get parameter for GET/POST
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

        # Step 1: Text generation using 2.5 Pro
        text_model = genai.GenerativeModel(TEXT_MODEL)
        text_resp = text_model.generate_content(prompt)
        text = text_resp.text.strip()

        # Step 2: Voice synthesis using light (1.5 Flash TTS)
        tts_model = genai.GenerativeModel(TTS_MODEL)
        audio_resp = tts_model.generate_content(
            [text],
            generation_config={"response_mime_type": "audio/wav"}
        )

        # Step 3: Encode audio to Base64
        audio_base64 = base64.b64encode(audio_resp.audio).decode("utf-8")

        return jsonify({
            "text": text,
            "audio_base64": audio_base64,
            "mime_type": "audio/wav",
            "text_model": TEXT_MODEL,
            "tts_model": TTS_MODEL
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

        stt_model = genai.GenerativeModel(TTS_MODEL)
        response = stt_model.generate_content(
            [audio_bytes],
            generation_config={"response_mime_type": "text/plain"}
        )

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
            "voice": TTS_MODEL
        },
        "endpoints": {
            "GET/POST /api/text": "Text generation (Gemini 2.5 Pro)",
            "GET/POST /api/voice": "Text + Voice output (Pro + Flash-TTS)",
            "POST /api/voice/transcribe": "Speech-to-text"
        }
    })


# ------------------ ENTRY ------------------ #
if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port)
