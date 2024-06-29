# src/build_vocab.py

import json
import os
from collections import Counter

def build_vocab(preprocessed_data_path, vocab_path):
    counter = Counter()
    with open(preprocessed_data_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
        for item in data['questions'] + data['contexts'] + data['answers']:
            tokens = item.split()  # Simple space tokenization
            counter.update(tokens)
    
    vocab = {word: idx for idx, (word, _) in enumerate(counter.most_common(), start=1)}
    vocab['<pad>'] = 0  # Add padding token
    
    with open(vocab_path, 'w', encoding='utf-8') as f:
        json.dump(vocab, f, indent=4)

    return vocab

if __name__ == "__main__":
    preprocessed_data_path = r'D:\Aarambh\data\preprocessed_data.json'
    vocab_path = r'D:\Aarambh\models\aarambh_vocab.json'

    build_vocab(preprocessed_data_path, vocab_path)
    print(f"Vocabulary built and saved to {vocab_path}")
