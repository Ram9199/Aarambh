import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), 'src')))

from aarambh.aarambh_wrapper import AarambhWrapper
from text_generation.aarambh_text_generation_wrapper import AarambhTextGenerationWrapper
from empathy.aarambh_empathy_wrapper import AarambhEmpathyWrapper
from translation.translation import TranslationModel
from image_recognition.image_recognition import ImageRecognitionModel
from voice_recognition.voice_recognition import LiveVoiceRecognition

# Initialize models
aarambh = AarambhWrapper(
    vocab_size=50257,
    d_model=512,
    nhead=8,
    num_encoder_layers=6,
    num_decoder_layers=6,
    dim_feedforward=2048,
    max_seq_length=512
)

aarambh_text_gen = AarambhTextGenerationWrapper(
    vocab_size=50257,
    d_model=512,
    nhead=8,
    num_encoder_layers=6,
    num_decoder_layers=6,
    dim_feedforward=2048,
    max_seq_length=512
)

aarambh_empathy = AarambhEmpathyWrapper(
    vocab_size=50257,
    d_model=512,
    nhead=8,
    num_encoder_layers=6,
    num_decoder_layers=6,
    dim_feedforward=2048,
    max_seq_length=512
)

# Load trained models
aarambh.load("../models/aarambh_model.pth", "../models/aarambh_vocab.json")
aarambh_text_gen.load("../models/aarambh_text_gen_model.pth", "../models/aarambh_text_gen_vocab.json")
aarambh_empathy.load("../models/aarambh_empathy_model.pth", "../models/aarambh_empathy_vocab.json")

# Initialize other models
translator = TranslationModel(model_name='Helsinki-NLP/opus-mt-en-de')
img_recog = ImageRecognitionModel()
voice_recog = LiveVoiceRecognition()

# Example usage
print("Aarambh Response:", aarambh.generate_response("Hello, how are you?"))
print("Generated Text:", aarambh_text_gen.generate_text("Once upon a time"))
print("Translation:", translator.translate("Hello, how are you?"))
print("Empathetic Response:", aarambh_empathy.generate_empathetic_response("I'm feeling sad."))
print("Image Text Recognition:", img_recog.recognize_text(r"D:\Aarambh\src\image_recognition\img.png"))

# Perform live speech recognition
print("Voice Recognition (live):", voice_recog.recognize_speech())

# Perform live emotion recognition
print("Emotion Recognition (live):", voice_recog.recognize_emotion())
