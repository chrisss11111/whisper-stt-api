from flask import Flask, request, jsonify
from faster_whisper import WhisperModel
import tempfile
import os

app = Flask(__name__)
model = WhisperModel("base", compute_type="int8")  # Choose "tiny", "base", "small", etc.

@app.route('/transcribe', methods=['POST'])
def transcribe():
    if 'file' not in request.files:
        return jsonify({"error": "No audio file uploaded"}), 400

    file = request.files['file']
    with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as temp_audio:
        file.save(temp_audio.name)
        segments, info = model.transcribe(temp_audio.name)
        text = "".join([seg.text for seg in segments])
        os.remove(temp_audio.name)

    return jsonify({"text": text.strip()})
