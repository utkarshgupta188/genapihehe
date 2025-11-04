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
        model_name = get_param("model", "models/gemini-2.5-flash")

        if not prompt:
            return jsonify({"error": "Missing prompt"}), 400

        model = genai.GenerativeModel(model_name)
        response = model.generate_content(prompt)

        return jsonify({
            "response": response.text.strip(),
            "model": model_name
        })
    except Exception as e:
        return jsonify({"error": str(e)}), 500


# ------------------ VOICE ENDPOINT ------------------ #
@app.route("/api/voice", methods=["GET", "POST"])
def handle_voice():
    try:
        prompt = get_param("prompt")
        model_name = get_param("model", "models/gemini-2.5-flash")

        if not prompt:
            return jsonify({"error": "Missing prompt"}), 400

        # Step 1: Generate AI text response
        text_model = genai.GenerativeModel(model_name)
        text_response = text_model.generate_content(prompt)
        text_output = text_response.text.strip()

        # Step 2: Convert text to speech using Gemini 2.5 Flash TTS
        tts_model = genai.GenerativeModel("models/gemini-2.5-flash-tts")
        audio_response = tts_model.generate_content(
            [text_output],
            generation_config={"response_mime_type": "audio/wav"}
        )

        # Step 3: Encode WAV audio as base64
        audio_base64 = base64.b64encode(audio_response.audio).decode("utf-8")

        return jsonify({
            "text": text_output,
            "audio_base64": audio_base64,
            "mime_type": "audio/wav",
            "model": model_name
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

        stt_model = genai.GenerativeModel("models/gemini-2.5-flash-tts")
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
            "default_text": "models/gemini-2.5-flash",
            "default_tts": "models/gemini-2.5-flash-tts",
            "optional": "models/gemini-2.5-pro"
        },
        "endpoints": {
            "GET/POST /api/text": "Text generation",
            "GET/POST /api/voice": "Text + Voice generation",
            "POST /api/voice/transcribe": "Speech-to-text"
        }
    })


# ------------------ ENTRY ------------------ #
if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port)
