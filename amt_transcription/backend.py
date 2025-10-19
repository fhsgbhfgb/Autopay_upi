"""
Flask Backend for Voice Payment Recognition using OpenAI Whisper

Installation:
    pip install openai-whisper flask flask-cors

Usage:
    python app.py
    Then open your HTML file in browser
"""

from flask import Flask, request, jsonify
from flask_cors import CORS
import whisper
import tempfile
import os
import re

app = Flask(__name__)
CORS(app)  # Enable CORS for HTML to communicate with this server

# Load Whisper model once at startup
print("Loading OpenAI Whisper model...")
model = whisper.load_model("base", device="cpu")
print("Model loaded successfully!")

def extract_amount(text):
    """
    Extract numeric amount from transcribed text
    
    Args:
        text: Transcribed text
        
    Returns:
        Extracted amount as integer or None
    """
    # First, try to find direct digits
    digits = re.findall(r'\d+', text)
    if digits:
        return int(digits[0])
    
    # Word to number conversion
    words_to_numbers = {
        'zero': 0, 'one': 1, 'two': 2, 'three': 3, 'four': 4,
        'five': 5, 'six': 6, 'seven': 7, 'eight': 8, 'nine': 9,
        'ten': 10, 'eleven': 11, 'twelve': 12, 'thirteen': 13,
        'fourteen': 14, 'fifteen': 15, 'sixteen': 16, 'seventeen': 17,
        'eighteen': 18, 'nineteen': 19, 'twenty': 20, 'thirty': 30,
        'forty': 40, 'fifty': 50, 'sixty': 60, 'seventy': 70,
        'eighty': 80, 'ninety': 90, 'hundred': 100, 'thousand': 1000
    }
    
    total = 0
    current = 0
    words = text.lower().split()
    
    for word in words:
        # Remove punctuation
        word = re.sub(r'[^\w\s]', '', word)
        
        if word in words_to_numbers:
            num = words_to_numbers[word]
            
            if num == 100:
                current = current * 100 if current else 100
            elif num == 1000:
                current = (current if current else 1) * 1000
                total += current
                current = 0
            else:
                current += num
    
    total += current
    return total if total > 0 else None

@app.route('/transcribe', methods=['POST'])
def transcribe_audio():
    """
    API endpoint to receive audio and return transcribed amount
    
    Expects:
        Audio file in request.files['audio']
        
    Returns:
        JSON with amount and transcribed text
    """
    try:
        if 'audio' not in request.files:
            return jsonify({'error': 'No audio file provided'}), 400
        
        audio_file = request.files['audio']
        
        # Save to temporary file
        with tempfile.NamedTemporaryFile(delete=False, suffix='.wav') as temp_audio:
            audio_file.save(temp_audio.name)
            temp_path = temp_audio.name
        
        try:
            # Transcribe using Whisper
            print("Transcribing audio...")
            result = model.transcribe(temp_path, language="en")
            transcribed_text = result["text"]
            print(f"Transcribed: {transcribed_text}")
            
            # Extract amount
            amount = extract_amount(transcribed_text)
            
            response = {
                'success': True,
                'text': transcribed_text,
                'amount': amount
            }
            
            return jsonify(response)
            
        finally:
            # Clean up temporary file
            os.unlink(temp_path)
    
    except Exception as e:
        print(f"Error: {str(e)}")
        return jsonify({'error': str(e)}), 500

@app.route('/health', methods=['GET'])
def health_check():
    """Health check endpoint"""
    return jsonify({'status': 'ok', 'model': 'whisper-base'})

if __name__ == '__main__':
    print("\n" + "="*50)
    print("Flask Server Running on http://localhost:5000")
    print("Whisper model ready for speech recognition")
    print("="*50 + "\n")
    app.run(debug=True, port=5000)
