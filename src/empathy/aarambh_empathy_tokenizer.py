import re
import json

class AarambhTokenizer:
    def __init__(self, vocab_file=None):
        if vocab_file:
            with open(vocab_file, 'r') as f:
                self.vocab = json.load(f)
            self.inv_vocab = {v: k for k, v in self.vocab.items()}
        else:
            self.vocab = {}
            self.inv_vocab = {}
        self.special_tokens = {"<pad>": 0, "<unk>": 1, "<bos>": 2, "<eos>": 3}

    def build_vocab(self, texts):
        words = set()
        for text in texts:
            words.update(re.findall(r'\w+', text.lower()))
        for i, word in enumerate(words, len(self.special_tokens)):
            self.vocab[word] = i
        self.vocab.update(self.special_tokens)
        self.inv_vocab = {v: k for k, v in self.vocab.items()}

    def tokenize(self, text):
        tokens = [self.vocab.get(word, self.vocab["<unk>"]) for word in re.findall(r'\w+', text.lower())]
        return [self.vocab["<bos>"]] + tokens + [self.vocab["<eos>"]]

    def detokenize(self, tokens):
        words = [self.inv_vocab.get(token, "<unk>") for token in tokens if token not in self.special_tokens.values()]
        return ' '.join(words)

    def save_vocab(self, vocab_file):
        with open(vocab_file, 'w') as f:
            json.dump(self.vocab, f)

    def load_vocab(self, vocab_file):
        with open(vocab_file, 'r') as f:
            self.vocab = json.load(f)
        self.inv_vocab = {v: k for k, v in self.vocab.items()}
