"use client";

import { useState, FormEvent, ChangeEvent } from "react";
import styled, { keyframes, ThemeProvider, DefaultTheme } from "styled-components";
import { FaSun, FaMoon, FaMicrophone, FaPaperPlane, FaFileUpload } from "react-icons/fa";

const lightTheme: DefaultTheme = {
  background: "#f0f0f0",
  text: "#000000",
  buttonBackground: "#0070f3",
  buttonText: "#ffffff",
  inputBackground: "#ffffff",
  inputText: "#000000",
  borderColor: "#ddd",
  hoverBackground: "#005bb5",
};

const darkTheme: DefaultTheme = {
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
  height: 100vh;
  position: relative;
`;

const Header = styled.div`
  display: flex;
  justify-content: space-between;
  width: 100%;
  align-items: center;
`;

const colorChange = keyframes`
  0% { color: red; }
  33% { color: green; }
  66% { color: blue; }
  100% { color: red; }
`;

const Title = styled.h1`
  font-size: 36px;
  margin-bottom: 20px;
  animation: ${colorChange} 5s infinite;
  background: linear-gradient(45deg, #f3ec78, #af4261, #0070f3);
  -webkit-background-clip: text;
  -webkit-text-fill-color: transparent;
`;

const Form = styled.form`
  display: flex;
  align-items: center;
  gap: 10px;
  width: 100%;
  position: absolute;
  bottom: 20px;
  background-color: ${(props) => props.theme.inputBackground};
  padding: 10px;
  border-radius: 25px;
  box-shadow: 0px 0px 10px rgba(0, 0, 0, 0.1);
`;

const Input = styled.input`
  padding: 10px;
  font-size: 16px;
  border: none;
  background-color: ${(props) => props.theme.inputBackground};
  color: ${(props) => props.theme.inputText};
  outline: none;
  flex-grow: 1;
  border-radius: 25px;
`;

const SubmitButton = styled.button`
  padding: 10px;
  font-size: 16px;
  cursor: pointer;
  background-color: ${(props) => props.theme.buttonBackground};
  color: ${(props) => props.theme.buttonText};
  border: none;
  border-radius: 50%;
  display: flex;
  align-items: center;
  justify-content: center;
  transition: background-color 0.3s;
  &:hover {
    background-color: ${(props) => props.theme.hoverBackground};
  }
`;

const ChatContainer = styled.div`
  width: 100%;
  max-height: calc(100vh - 150px);
  overflow-y: auto;
  background-color: ${(props) => props.theme.inputBackground};
  border: 1px solid ${(props) => props.theme.borderColor};
  border-radius: 4px;
  padding: 10px;
  margin-top: 20px;
`;

interface ChatBubbleProps {
  isUser: boolean;
}

const ChatBubble = styled.div<ChatBubbleProps>`
  background-color: ${(props) => (props.isUser ? "#005bb5" : "#333")};
  color: ${(props) => props.theme.buttonText};
  padding: 10px;
  border-radius: 10px;
  margin-bottom: 10px;
  max-width: 80%;
  ${(props) => (props.isUser ? "margin-left: auto;" : "margin-right: auto;")}
`;

const FileUpload = styled.input`
  display: none;
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
  const [chats, setChats] = useState<{ text: string; isUser: boolean }[]>([]);
  const [loading, setLoading] = useState<boolean>(false);
  const [theme, setTheme] = useState(lightTheme);
  const [mediaRecorder, setMediaRecorder] = useState<MediaRecorder | null>(null);

  const toggleTheme = () => {
    setTheme(theme === lightTheme ? darkTheme : lightTheme);
  };

  const handleSubmit = async (e: FormEvent) => {
    e.preventDefault();
    if (!prompt.trim()) return;

    setChats((prevChats) => [...prevChats, { text: prompt, isUser: true }]);
    setPrompt("");
    setLoading(true);

    let endpoint = "http://127.0.0.1:8000/generate/";

    if (prompt.toLowerCase().includes("translate")) {
      endpoint = "http://127.0.0.1:8000/translate/";
    }

    try {
      const res = await fetch(endpoint, {
        method: "POST",
        headers: {
          "Content-Type": "application/json",
        },
        body: JSON.stringify({ prompt }),
      });

      if (res.ok) {
        const data = await res.json();
        setChats((prevChats) => [
          ...prevChats,
          { text: data.response || data.translated_text, isUser: false },
        ]);
      } else {
        setChats((prevChats) => [
          ...prevChats,
          { text: "Error: Unable to fetch response.", isUser: false },
        ]);
      }
    } catch (error: unknown) {
      setChats((prevChats) => [
        ...prevChats,
        {
          text: error instanceof Error ? `Error: ${error.message}` : "Error: Unknown error occurred.",
          isUser: false,
        },
      ]);
    }
    setLoading(false);
  };

