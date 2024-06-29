import os
import sys
import torch
from fastapi import FastAPI, HTTPException, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import logging
import speech_recognition as sr
from io import BytesIO
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Ensure the current directory is in the system path
current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(current_dir)

from aarambh.aarambh_wrapper import AarambhWrapper
from translation.translation import TranslationModel
from image_recognition.image_recognition import ImageRecognitionModel

def initialize_aarambh():
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
        return aarambh
    except Exception as e:
        logger.error(f"Failed to load Aarambh model: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to load Aarambh model: {e}")

def initialize_translator():
    try:
        return TranslationModel(model_name='Helsinki-NLP/opus-mt-en-de')
    except Exception as e:
        logger.error(f"Failed to initialize translation model: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to initialize translation model: {e}")

def initialize_image_recognition():
    try:
        return ImageRecognitionModel()
    except Exception as e:
        logger.error(f"Failed to initialize image recognition model: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to initialize image recognition model: {e}")

# Initialize models
aarambh = initialize_aarambh()
translator = initialize_translator()
img_recog = initialize_image_recognition()

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
async def recognize_speech(file: UploadFile = File(...)):
    try:
        contents = await file.read()
        recognizer = sr.Recognizer()
        audio_file = BytesIO(contents)
        with sr.AudioFile(audio_file) as source:
            audio_data = recognizer.record(source)
            try:
                text = recognizer.recognize_google(audio_data)
                return {"recognized_speech": text}
            except sr.UnknownValueError:
                return {"recognized_speech": "Google Speech Recognition could not understand audio"}
            except sr.RequestError as e:
                return {"recognized_speech": f"Could not request results from Google Speech Recognition service; {e}"}
    except Exception as e:
        logger.error(f"Error recognizing speech: {e}")
        raise HTTPException(status_code=500, detail=f"Error recognizing speech: {e}")

@app.post("/voice_recognition/emotion/")
async def recognize_emotion(file: UploadFile = File(...)):
    # Placeholder for emotion recognition logic using live audio
    # Implement your custom emotion recognition logic here
    return {"recognized_emotion": "Emotion recognition is not implemented yet"}

@app.post("/upload/")
async def upload_file(file: UploadFile = File(...)):
    try:
        contents = await file.read()
        # Implement your file processing logic here
        return {"filename": file.filename, "content_type": file.content_type}
    except Exception as e:
        logger.error(f"Error uploading file: {e}")
        raise HTTPException(status_code=500, detail=f"Error uploading file: {e}")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)