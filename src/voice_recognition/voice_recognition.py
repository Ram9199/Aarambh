import os
import pyaudio
import torch                                                                                                     
import numpy as np
from transformers import Wav2Vec2Processor, Wav2Vec2ForCTC, pipeline, AutoProcessor, AutoModelForAudioClassification

class LiveVoiceRecognition:
    def __init__(self):
        # Define the paths for the local models relative to the script location
        self.model_dir_sr = os.path.join(os.path.dirname(__file__), "models/wav2vec2-large-xlsr-53")
        self.model_dir_em = os.path.join(os.path.dirname(__file__), "models/wav2vec2-lg-xlsr-en-speech-emotion-recognition")

       
        # Load the processor and model for speech recognition from local paths
        self.processor_sr = Wav2Vec2Processor.from_pretrained(self.model_dir_sr)
        self.model_sr = Wav2Vec2ForCTC.from_pretrained(self.model_dir_sr)

        # Load the processor and model for emotion recognition from local paths
        self.processor_em = AutoProcessor.from_pretrained(self.model_dir_em)
        self.model_em = AutoModelForAudioClassification.from_pretrained(self.model_dir_em)
        self.emotion_pipeline = pipeline("audio-classification", model=self.model_em, processor=self.processor_em)

        # Audio parameters
        self.sample_rate = 16000
        self.chunk_size = 1024

    def verify_model_directory(self, directory, required_files):
        if not os.path.exists(directory):
            raise FileNotFoundError(f"Model directory {directory} does not exist")
        for required_file in required_files:
            if not os.path.isfile(os.path.join(directory, required_file)):
                raise FileNotFoundError(f"Required file {required_file} not found in directory {directory}")

    def recognize_speech(self):
        p = pyaudio.PyAudio()
        stream = p.open(format=pyaudio.paInt16,
                        channels=1,
                        rate=self.sample_rate,
                        input=True,
                        frames_per_buffer=self.chunk_size)

        print("Please say something...")

        audio_data = []
        while True:
            data = stream.read(self.chunk_size)
            audio_np = np.frombuffer(data, dtype=np.int16).astype(np.float32) / 32768.0
            audio_data.extend(audio_np)

            # Capture 5 seconds of audio
            if len(audio_data) > self.sample_rate * 5:
                break

        stream.stop_stream()
        stream.close()
        p.terminate()

        audio_data = np.array(audio_data)
        input_values = self.processor_sr(audio_data, sampling_rate=self.sample_rate, return_tensors="pt").input_values

        with torch.no_grad():
            logits = self.model_sr(input_values).logits

        predicted_ids = torch.argmax(logits, dim=-1)
        transcription = self.processor_sr.decode(predicted_ids[0])

        return transcription

    def recognize_emotion(self):
        p = pyaudio.PyAudio()
        stream = p.open(format=pyaudio.paInt16,
                        channels=1,
                        rate=self.sample_rate,
                        input=True,
                        frames_per_buffer=self.chunk_size)

        print("Please say something...")

        audio_data = []
        while True:
            data = stream.read(self.chunk_size)
            audio_np = np.frombuffer(data, dtype=np.int16).astype(np.float32) / 32768.0
            audio_data.extend(audio_np)

            # Capture 5 seconds of audio
            if len(audio_data) > self.sample_rate * 5:
                break

        stream.stop_stream()
        stream.close()
        p.terminate()

        audio_data = np.array(audio_data)

        # Emotion recognition
        emotions = self.emotion_pipeline(audio_data, sampling_rate=self.sample_rate)
        return emotions

if __name__ == "__main__":
    recognizer = LiveVoiceRecognition()
    text = recognizer.recognize_speech()
    print("Recognized text:", text)

    emotions = recognizer.recognize_emotion()
    print("Recognized emotions:", emotions)
