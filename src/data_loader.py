# src/data_loader.py

import json
import os
import torch
from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pad_sequence
from collections import Counter

class CustomDataset(Dataset):
    def __init__(self, preprocessed_data_path):
        with open(preprocessed_data_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        self.questions = data['questions']
        self.contexts = data['contexts']
        self.answers = data['answers']

    def __len__(self):
        return len(self.questions)

    def __getitem__(self, idx):
        return {
            'question': self.questions[idx],
            'context': self.contexts[idx],
            'answer': self.answers[idx]
        }

def pad_collate(batch):
    questions = [torch.tensor(item['question']) for item in batch]
    contexts = [torch.tensor(item['context']) for item in batch]
    answers = [torch.tensor(item['answer']) for item in batch]
    
    padded_questions = pad_sequence(questions, batch_first=True, padding_value=0)
    padded_contexts = pad_sequence(contexts, batch_first=True, padding_value=0)
    padded_answers = pad_sequence(answers, batch_first=True, padding_value=0)
    
    combined_inputs = torch.cat((padded_questions, padded_contexts), dim=1)
    
    return {'combined_inputs': combined_inputs, 'answers': padded_answers}

def create_dataloader(preprocessed_data_path, batch_size=2):
    dataset = CustomDataset(preprocessed_data_path)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, collate_fn=pad_collate)
    return dataloader

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
