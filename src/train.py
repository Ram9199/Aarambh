import sys
import os
import torch
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), 'src')))

from aarambh.aarambh_wrapper import AarambhWrapper
from text_generation.aarambh_text_generation_wrapper import AarambhTextGenerationWrapper
from empathy.aarambh_empathy_wrapper import AarambhEmpathyWrapper
from data.dataset import DummyDataset

if __name__ == "__main__":
    # Initialize Aarambh model with specified parameters
    aarambh = AarambhWrapper(
        vocab_size=50257,
        d_model=512,
        nhead=8,
        num_encoder_layers=6,
        num_decoder_layers=6,
        dim_feedforward=2048,
        max_seq_length=512
    )
    
    # Initialize Aarambh text generation model with specified parameters
    aarambh_text_gen = AarambhTextGenerationWrapper(
        vocab_size=50257,
        d_model=512,
        nhead=8,
        num_encoder_layers=6,
        num_decoder_layers=6,
        dim_feedforward=2048,
        max_seq_length=512
    )
    
    # Initialize Aarambh empathy model with specified parameters
    aarambh_empathy = AarambhEmpathyWrapper(
        vocab_size=50257,
        d_model=512,
        nhead=8,
        num_encoder_layers=6,
        num_decoder_layers=6,
        dim_feedforward=2048,
        max_seq_length=512
    )
    
    # Define your training texts with more diverse examples
    texts = [
        "Hello, how are you?", 
        "I am a chatbot.", 
        "What can you do?", 
        "I can chat with you.", 
        "Tell me a joke.",
        "What's the weather like today?",
        "Can you help me with my homework?",
        "I love programming.",
        "Do you know any good recipes?",
        "How do you make a chatbot?"
    ]
    dataset = DummyDataset(texts)
    
    # Build vocabulary from dataset
    aarambh.tokenizer.build_vocab(texts)
    aarambh_text_gen.tokenizer.build_vocab(texts)
    aarambh_empathy.tokenizer.build_vocab(texts)
    
    # Ensure the models directory exists
    os.makedirs("../models", exist_ok=True)
    
    # Train the Aarambh model
    print("Training Aarambh model...")
    aarambh.train(dataset, epochs=10, batch_size=2, lr=5e-5)
    aarambh.save("../models/aarambh_model.pth", "../models/aarambh_vocab.json")

    # Train the Aarambh text generation model
    print("Training Aarambh text generation model...")
    aarambh_text_gen.train(dataset, epochs=10, batch_size=2, lr=5e-5)
    aarambh_text_gen.save("../models/aarambh_text_gen_model.pth", "../models/aarambh_text_gen_vocab.json")

    # Train the Aarambh empathy model
    print("Training Aarambh empathy model...")
    aarambh_empathy.train(dataset, epochs=10, batch_size=2, lr=5e-5)
    aarambh_empathy.save("../models/aarambh_empathy_model.pth", "../models/aarambh_empathy_vocab.json")
