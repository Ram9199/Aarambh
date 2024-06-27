import json

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

    def save_vocab(self, vocab_path):
        with open(vocab_path, 'w') as f:
            json.dump(self.vocab, f)

    def load_vocab(self, vocab_path):
        with open(vocab_path, 'r') as f:
            self.vocab = json.load(f)
        self.reverse_vocab = {idx: token for token, idx in self.vocab.items()}
        self.vocab_size = len(self.vocab)
