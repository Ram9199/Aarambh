import pyaudio
import wave
import speech_recognition as sr

class LiveVoiceRecognition:
    def __init__(self, chunk=1024, format=pyaudio.paInt16, channels=1, rate=44100, duration=5):
        self.chunk = chunk
        self.format = format
        self.channels = channels
        self.rate = rate
        self.duration = duration

    def record_audio(self, filename="live_audio.wav"):
        p = pyaudio.PyAudio()
        stream = p.open(format=self.format,
                        channels=self.channels,
                        rate=self.rate,
                        input=True,
                        frames_per_buffer=self.chunk)
        print("Recording...")
        frames = []

        for _ in range(0, int(self.rate / self.chunk * self.duration)):
            data = stream.read(self.chunk)
            frames.append(data)

        print("Recording finished")

        stream.stop_stream()
        stream.close()
        p.terminate()

        wf = wave.open(filename, 'wb')
        wf.setnchannels(self.channels)
        wf.setsampwidth(p.get_sample_size(self.format))
        wf.setframerate(self.rate)
        wf.writeframes(b''.join(frames))
        wf.close()
        return filename

    def recognize_speech(self, filename="live_audio.wav"):
        recognizer = sr.Recognizer()
        with sr.AudioFile(filename) as source:
            audio_data = source.record()
            try:
                text = recognizer.recognize_google(audio_data)
                return text
            except sr.UnknownValueError:
                return "Google Speech Recognition could not understand audio"
            except sr.RequestError as e:
                return f"Could not request results from Google Speech Recognition service; {e}"

    def recognize_emotion(self, filename="live_audio.wav"):
        # Placeholder for emotion recognition logic using live audio
        # Implement your custom emotion recognition logic here
        return "Emotion recognition is not implemented yet"

# Usage example
if __name__ == "__main__":
    voice_recog = LiveVoiceRecognition()
    audio_file = voice_recog.record_audio()
    try:
        print("Voice Recognition (live):", voice_recog.recognize_speech(audio_file))
        print("Emotion Recognition (live):", voice_recog.recognize_emotion(audio_file))
    except Exception as e:
        print(f"Error during recognition: {e}")
