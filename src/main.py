import os
import sys
import torch
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import logging

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Ensure the current directory is in the system path
current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(current_dir)

from aarambh.aarambh_wrapper import AarambhWrapper
from translation.translation import TranslationModel
from image_recognition.image_recognition import ImageRecognitionModel
from voice_recognition.voice_recognition import LiveVoiceRecognition

# Initialize models
try:
    aarambh = AarambhWrapper(
        vocab_size=50257,
        d_model=512,
        nhead=8,
        num_encoder_layers=6,
        num_decoder_layers=6,
        dim_feedforward=2048,
        max_seq_length=5000
    )
    aarambh.load(os.path.join(current_dir, "..", "models", "aarambh_model.pth"))
except Exception as e:
    logger.error(f"Failed to load Aarambh model: {e}")
    raise HTTPException(status_code=500, detail=f"Failed to load Aarambh model: {e}")

try:
    translator = TranslationModel(model_name='Helsinki-NLP/opus-mt-en-de')
except Exception as e:
    logger.error(f"Failed to initialize translation model: {e}")
    raise HTTPException(status_code=500, detail=f"Failed to initialize translation model: {e}")

try:
    img_recog = ImageRecognitionModel()
except Exception as e:
    logger.error(f"Failed to initialize image recognition model: {e}")
    raise HTTPException(status_code=500, detail=f"Failed to initialize image recognition model: {e}")

try:
    voice_recog = LiveVoiceRecognition()
except Exception as e:
    logger.error(f"Failed to initialize voice recognition model: {e}")
    raise HTTPException(status_code=500, detail=f"Failed to initialize voice recognition model: {e}")

# FastAPI app
app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allows all origins
    allow_credentials=True,
    allow_methods=["*"],  # Allows all methods
    allow_headers=["*"],  # Allows all headers
)

class QueryRequest(BaseModel):
    prompt: str

class TranslateRequest(BaseModel):
    text: str
    target_language: str

class ImageRecognitionRequest(BaseModel):
    image_path: str

@app.get("/")
def read_root():
    return {"message": "Welcome to the Aarambh API"}

@app.post("/generate/")
def generate_response(request: QueryRequest):
    try:
        response = aarambh.generate_response(request.prompt)
        return {"response": response}
    except Exception as e:
        logger.error(f"Error generating response: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/translate/")
def translate_text(request: TranslateRequest):
    try:
        translated_text = translator.translate(request.text, request.target_language)
        return {"translated_text": translated_text}
    except Exception as e:
        logger.error(f"Error translating text: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/image_recognition/")
def recognize_text_in_image(request: ImageRecognitionRequest):
    try:
        text = img_recog.recognize_text(request.image_path)
        return {"recognized_text": text}
    except Exception as e:
        logger.error(f"Error recognizing text in image: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/voice_recognition/speech/")
def recognize_speech():
    try:
        audio_file = voice_recog.record_audio()
        text = voice_recog.recognize_speech(audio_file)
        return {"recognized_speech": text}
    except Exception as e:
        logger.error(f"Error recognizing speech: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/voice_recognition/emotion/")
def recognize_emotion():
    try:
        audio_file = voice_recog.record_audio()
        emotion = voice_recog.recognize_emotion(audio_file)
        return {"recognized_emotion": emotion}
    except Exception as e:
        logger.error(f"Error recognizing emotion: {e}")
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
