import os
import json
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("TOKENIZER.log")

file_handler = logging.FileHandler("TOKENIZER.log")
file_handler.setLevel(logging.INFO)
formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
file_handler.setFormatter(formatter)
logger.addHandler(file_handler)

class AarambhTokenizer:
    def __init__(self):
        self.vocab = {}
        self.reverse_vocab = {}
        self.vocab_size = 0

    def build_vocab(self, texts):
        unique_tokens = set()
        for text in texts:
            tokens = text.split()
            unique_tokens.update(tokens)
        
        self.vocab = {token: idx for idx, token in enumerate(unique_tokens, start=2)}
        self.vocab["<pad>"] = 0
        self.vocab["<unk>"] = 1
        self.reverse_vocab = {idx: token for token, idx in self.vocab.items()}
        self.vocab_size = len(self.vocab)

    def tokenize(self, text):
        return [self.vocab.get(token, self.vocab["<unk>"]) for token in text.split()]

    def detokenize(self, tokens):
        return ' '.join([self.reverse_vocab.get(token, "<unk>") for token in tokens])

    def save_vocab(self, vocab_filename):
        vocab_dir = os.path.join('d:', 'Aarambh', 'models')
        os.makedirs(vocab_dir, exist_ok=True)
        vocab_path = os.path.join(vocab_dir, vocab_filename)
        with open(vocab_path, 'w') as f:
            json.dump(self.vocab, f)
        print(f"Vocabulary saved to {vocab_path}")

    def load_vocab(self, vocab_filename):
        print(f"Initial vocab_filename: {vocab_filename}")
        vocab_path = os.path.join('d:', 'Aarambh', 'models', vocab_filename)
        print(f"Constructed vocab_path: {vocab_path}")
        try:
            with open(vocab_path, 'r') as f:
                vocab_data = json.load(f)
            self.vocab = vocab_data
            self.reverse_vocab = {idx: token for token, idx in self.vocab.items()}
            self.vocab_size = len(self.vocab)
            print(f"Vocabulary loaded from {vocab_path}")
        except FileNotFoundError as e:
            print(f"FileNotFoundError: {e}")
            raise
