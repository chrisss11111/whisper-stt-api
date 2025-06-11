from flask import Flask, request, jsonify, send_file
from faster_whisper import WhisperModel
import tempfile
import os
import requests
import openai

app = Flask(__name__)

# Load Whisper model
model = WhisperModel("base", compute_type="int8")

# Load environment variables
openai.api_key = os.getenv("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")
ELEVENLABS_API_KEY = os.getenv("sk_395493579726778a10d5f99773a5f40ddae1fcc9af309b1b")
ELEVENLABS_VOICE_ID = os.getenv("IRHApOXLvnW57QJPQH2P")

@app.route('/process', methods=['POST'])
def process_audio():
    if 'file' not in request.files:
        return jsonify({"error": "No file uploaded"}), 400

    file = request.files['file']
    with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as temp_audio:
        file.save(temp_audio.name)

        # Step 1: Transcribe
        segments, _ = model.transcribe(temp_audio.name)
        text = ''.join([seg.text for seg in segments]).strip()
        os.remove(temp_audio.name)

    # Step 2: Ask ChatGPT
    gpt_response = openai.ChatCompletion.create(
        model="gpt-3.5-turbo",
        messages=[{"role": "user", "content": text}]
    )
    reply = gpt_response.choices[0].message.content

    # Step 3: Generate speech with ElevenLabs
    tts_url = f"https://api.elevenlabs.io/v1/text-to-speech/{ELEVENLABS_VOICE_ID}"
    headers = {
        "xi-api-key": ELEVENLABS_API_KEY,
        "Content-Type": "application/json"
    }
    tts_data = {
        "text": reply,
        "model_id": "eleven_monolingual_v1",
        "voice_settings": {
            "stability": 0.5,
            "similarity_boost": 0.5
        }
    }

    tts_response = requests.post(tts_url, json=tts_data, headers=headers)

    if tts_response.status_code != 200:
        return jsonify({"error": "TTS failed"}), 500

    # Save mp3 to temp file and return it
    with tempfile.NamedTemporaryFile(delete=False, suffix=".mp3") as temp_mp3:
        temp_mp3.write(tts_response.content)
        temp_mp3.flush()
        return send_file(temp_mp3.name, mimetype="audio/mpeg")

if __name__ == '__main__':
    app.run(debug=True, host="0.0.0.0")
