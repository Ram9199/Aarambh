"use client";

import { useState, FormEvent } from "react";
import styled, { keyframes, ThemeProvider } from "styled-components";
import { FaSun, FaMoon } from "react-icons/fa";
import styles from "./Home.module.css";

const lightTheme = {
  background: "#f0f0f0",
  text: "#000000",
  buttonBackground: "#0070f3",
  buttonText: "#ffffff",
  inputBackground: "#ffffff",
  inputText: "#000000",
  borderColor: "#ddd",
  hoverBackground: "#005bb5",
};

const darkTheme = {
  background: "#1c1c1c",
  text: "#ffffff",
  buttonBackground: "#0070f3",
  buttonText: "#ffffff",
  inputBackground: "#333333",
  inputText: "#ffffff",
  borderColor: "#555555",
  hoverBackground: "#005bb5",
};

const Container = styled.div`
  padding: 20px;
  max-width: 800px;
  margin: 0 auto;
  display: flex;
  flex-direction: column;
  align-items: center;
  background-color: ${(props) => props.theme.background};
  color: ${(props) => props.theme.text};
  font-family: "Roboto", sans-serif;
  transition: all 0.3s ease;
`;

const Header = styled.div`
  display: flex;
  justify-content: space-between;
  width: 100%;
`;

const Title = styled.h1`
  font-size: 36px;
  margin-bottom: 20px;
  animation: ${() => colorChange} 5s infinite;
  background: linear-gradient(45deg, #f3ec78, #af4261, #0070f3);
  -webkit-background-clip: text;
  -webkit-text-fill-color: transparent;
`;

const Form = styled.form`
  display: flex;
  flex-direction: column;
  gap: 10px;
  width: 100%;
`;

const Input = styled.input`
  padding: 10px;
  font-size: 16px;
  border: 2px solid ${(props) => props.theme.borderColor};
  background-color: ${(props) => props.theme.inputBackground};
  color: ${(props) => props.theme.inputText};
  border-radius: 4px;
  outline: none;
  transition: border-color 0.3s;
  &:focus {
    border-color: #0070f3;
  }
`;

const Button = styled.button`
  padding: 10px;
  font-size: 16px;
  cursor: pointer;
  background-color: ${(props) => props.theme.buttonBackground};
  color: ${(props) => props.theme.buttonText};
  border: none;
  border-radius: 4px;
  transition: background-color 0.3s, transform 0.2s;
  &:hover {
    background-color: ${(props) => props.theme.hoverBackground};
    transform: scale(1.05);
  }
`;

const Response = styled.p`
  margin-top: 20px;
  font-size: 18px;
`;

const colorChange = keyframes`
  0% { color: red; }
  33% { color: green; }
  66% { color: blue; }
  100% { color: red; }
`;

const TypingAnimation = keyframes`
  0% { opacity: 1; }
  50% { opacity: 0.5; }
  100% { opacity: 1; }
`;

const Loading = styled.span`
  font-size: 18px;
  animation: ${TypingAnimation} 1s infinite;
`;

const ToggleButton = styled.button`
  background: none;
  border: none;
  cursor: pointer;
  font-size: 24px;
  color: ${(props) => props.theme.text};
  outline: none;
  transition: color 0.3s;
`;

export default function Home() {
  const [prompt, setPrompt] = useState<string>("");
  const [response, setResponse] = useState<string>("");
  const [translatedText, setTranslatedText] = useState<string>("");
  const [recognizedText, setRecognizedText] = useState<string>("");
  const [recognizedSpeech, setRecognizedSpeech] = useState<string>("");
  const [recognizedEmotion, setRecognizedEmotion] = useState<string>("");
  const [loading, setLoading] = useState<boolean>(false);
  const [theme, setTheme] = useState(lightTheme);

  const toggleTheme = () => {
    setTheme(theme === lightTheme ? darkTheme : lightTheme);
  };

  const handleSubmit = async (e: FormEvent) => {
    e.preventDefault();
    setLoading(true);
    try {
      const res = await fetch("http://127.0.0.1:8000/generate/", {
        method: "POST",
        headers: {
          "Content-Type": "application/json",
        },
        body: JSON.stringify({ prompt }),
      });

      if (res.ok) {
        const data = await res.json();
        setResponse(data.response);
      } else {
        setResponse("Error: Unable to fetch response.");
      }
    } catch (error: unknown) {
      if (error instanceof Error) {
        setResponse(`Error: ${error.message}`);
      } else {
        setResponse("Error: Unknown error occurred.");
      }
    }
    setLoading(false);
  };

  const handleTranslate = async (text: string, targetLanguage: string) => {
    setLoading(true);
    try {
      const res = await fetch("http://127.0.0.1:8000/translate/", {
        method: "POST",
        headers: {
          "Content-Type": "application/json",
        },
        body: JSON.stringify({ text, target_language: targetLanguage }),
      });

      if (res.ok) {
        const data = await res.json();
        setTranslatedText(data.translated_text);
      } else {
        setTranslatedText("Error: Unable to translate text.");
      }
    } catch (error: unknown) {
      if (error instanceof Error) {
        setTranslatedText(`Error: ${error.message}`);
      } else {
        setTranslatedText("Error: Unknown error occurred.");
      }
    }
    setLoading(false);
  };

  const handleImageRecognition = async (imagePath: string) => {
    setLoading(true);
    try {
      const res = await fetch("http://127.0.0.1:8000/image_recognition/", {
        method: "POST",
        headers: {
          "Content-Type": "application/json",
        },
        body: JSON.stringify({ image_path: imagePath }),
      });

      if (res.ok) {
        const data = await res.json();
        setRecognizedText(data.recognized_text);
      } else {
        setRecognizedText("Error: Unable to recognize text.");
      }
    } catch (error: unknown) {
      if (error instanceof Error) {
        setRecognizedText(`Error: ${error.message}`);
      } else {
        setRecognizedText("Error: Unknown error occurred.");
      }
    }
    setLoading(false);
  };

  const handleSpeechRecognition = async () => {
    setLoading(true);
    try {
      const res = await fetch(
        "http://127.0.0.1:8000/voice_recognition/speech/",
        {
          method: "POST",
          headers: {
            "Content-Type": "application/json",
          },
        }
      );

      if (res.ok) {
        const data = await res.json();
        setRecognizedSpeech(data.recognized_speech);
      } else {
        setRecognizedSpeech("Error: Unable to recognize speech.");
      }
    } catch (error: unknown) {
      if (error instanceof Error) {
        setRecognizedSpeech(`Error: ${error.message}`);
      } else {
        setRecognizedSpeech("Error: Unknown error occurred.");
      }
    }
    setLoading(false);
  };

  const handleEmotionRecognition = async () => {
    setLoading(true);
    try {
      const res = await fetch(
        "http://127.0.0.1:8000/voice_recognition/emotion/",
        {
          method: "POST",
          headers: {
            "Content-Type": "application/json",
          },
        }
      );

      if (res.ok) {
        const data = await res.json();
        setRecognizedEmotion(data.recognized_emotion);
      } else {
        setRecognizedEmotion("Error: Unable to recognize emotion.");
      }
    } catch (error: unknown) {
      if (error instanceof Error) {
        setRecognizedEmotion(`Error: ${error.message}`);
      } else {
        setRecognizedEmotion("Error: Unknown error occurred.");
      }
    }
    setLoading(false);
  };

  return (
    <ThemeProvider theme={theme}>
      <Container>
        <Header>
          <Title>AARAMBH</Title>
          <ToggleButton onClick={toggleTheme}>
            {theme === lightTheme ? <FaMoon /> : <FaSun />}
          </ToggleButton>
        </Header>
        <Form onSubmit={handleSubmit}>
          <Input
            type="text"
            value={prompt}
            onChange={(e) => setPrompt(e.target.value)}
            placeholder="Enter your prompt"
          />
          <Button type="submit">Submit</Button>
        </Form>
        {loading ? (
          <Loading>Typing...</Loading>
        ) : (
          <Response>Response: {response}</Response>
        )}

        <h2>Translate Text</h2>
        <Button onClick={() => handleTranslate("Hello", "es")}>
          Translate 'Hello' to Spanish
        </Button>
        <Response>Translated Text: {translatedText}</Response>

        <h2>Image Recognition</h2>
        <Button onClick={() => handleImageRecognition("path_to_image.png")}>
          Recognize Text in Image
        </Button>
        <Response>Recognized Text: {recognizedText}</Response>

        <h2>Speech Recognition</h2>
        <Button onClick={handleSpeechRecognition}>Recognize Speech</Button>
        <Response>Recognized Speech: {recognizedSpeech}</Response>

        <h2>Emotion Recognition</h2>
        <Button onClick={handleEmotionRecognition}>Recognize Emotion</Button>
        <Response>Recognized Emotion: {recognizedEmotion}</Response>
      </Container>
    </ThemeProvider>
  );
}
