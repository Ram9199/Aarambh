import os
import sys
import json
import torch
from fastapi import FastAPI, HTTPException, Form, UploadFile, File
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
import logging
from dotenv import load_dotenv
from io import BytesIO
import speech_recognition as sr
from .aarambh.aarambh_wrapper import AarambhWrapper
from src.data_loader import create_dataloader, build_vocab

# Load environment variables from .env file
load_dotenv()

# Setup logging
log_file_path = os.path.join(os.path.dirname(__file__), 'qa_model.log')
logging.basicConfig(
    filename=log_file_path,
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)

app = FastAPI()

# Add CORS middleware
origins = [
    "http://localhost",
    "http://localhost:3000",  # Next.js development server
    "http://localhost:8000",
    "https://your-deployment-url"  # Update with your deployment URL
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize Aarambh model and tokenizer
def initialize_aarambh(vocab_size):
    try:
        aarambh = AarambhWrapper(
            vocab_size=vocab_size,
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
        logging.error(f"Failed to load Aarambh model: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to load Aarambh model: {e}")

def initialize_dataloader_and_vocab():
    try:
        data_dir = os.path.abspath(os.path.join(current_dir, '..', 'data'))
        preprocessed_data_path = os.path.join(data_dir, 'preprocessed_data.json')
        vocab_path = os.path.join(current_dir, '..', 'models', 'aarambh_vocab.json')
        
        with open(vocab_path, 'r', encoding='utf-8') as f:
            vocab = json.load(f)
        
        vocab_size = len(vocab)
        
        dataloader = create_dataloader(preprocessed_data_path, batch_size=2)
        return dataloader, vocab_size
    except Exception as e:
        logging.error(f"Failed to create DataLoader and load vocabulary: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to create DataLoader and load vocabulary: {e}")

# Initialize models
dataloader, vocab_size = initialize_dataloader_and_vocab()
aarambh = initialize_aarambh(vocab_size)

@app.post("/generate/")
async def generate_response(question: str = Form(...), context: str = Form(...)):
    try:
        response = aarambh.generate_response(question, context)
        return JSONResponse(content={"response": response})
    except Exception as e:
        logging.error(f"Error generating response: {e}")
        return JSONResponse(content={"error": str(e)}, status_code=500)

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
