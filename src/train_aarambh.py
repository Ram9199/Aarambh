import os
import logging
import json
import torch
from torch.utils.data import Dataset, DataLoader
from torch.nn import CrossEntropyLoss
import torch.optim as optim
from sklearn.model_selection import train_test_split
import pandas as pd

# Add the src directory to the system path
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'src')))

from aarambh.aarambh_wrapper import AarambhWrapper
from aarambh.aarambh_tokenizer import AarambhTokenizer
from aarambh.aarambh import Aarambh

# Setup logging
logging.basicConfig(level=logging.INFO)

class QADataset(Dataset):
    def __init__(self, dataframe, tokenizer, max_length=512):
        self.questions = dataframe['questions'].values
        self.contexts = dataframe['contexts'].values
        self.answers = dataframe['answers'].values
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.questions)

    def __getitem__(self, idx):
        input_text = f"Question: {self.questions[idx]}\nContext: {self.contexts[idx]}\nAnswer: {self.answers[idx]}"
        tokens = self.tokenizer.tokenize(input_text)
        tokens = tokens[:self.max_length]  # Truncate if too long
        input_ids = torch.tensor(tokens, dtype=torch.long)
        return {'input_ids': input_ids, 'labels': input_ids.clone()}

def collate_fn(batch):
    max_len = max(len(item['input_ids']) for item in batch)

    padded_input_ids = []
    for item in batch:
        input_ids = item['input_ids']
        padded = torch.cat([input_ids, torch.zeros(max_len - len(input_ids), dtype=torch.long)])
        padded_input_ids.append(padded)

    return {'input_ids': torch.stack(padded_input_ids), 'labels': torch.stack(padded_input_ids)}

def train_model(dataframe):
    logging.info("Initializing tokenizer...")
    tokenizer = AarambhTokenizer()
    texts = dataframe['questions'].tolist() + dataframe['contexts'].tolist() + dataframe['answers'].tolist()
    tokenizer.build_vocab(texts)
    logging.info("Tokenizer initialized.")

    logging.info("Initializing model...")
    model_wrapper = AarambhWrapper(
        vocab_size=tokenizer.vocab_size,
        d_model=512,
        nhead=8,
        num_encoder_layers=6,
        num_decoder_layers=6,
        dim_feedforward=2048,
        max_seq_length=512
    )
    logging.info("Model initialized.")

    # Split the data into training and evaluation sets
    train_df, eval_df = train_test_split(dataframe, test_size=0.1)

    train_dataset = QADataset(train_df, tokenizer)
    eval_dataset = QADataset(eval_df, tokenizer)

    train_dataloader = DataLoader(train_dataset, batch_size=4, shuffle=True, collate_fn=collate_fn)
    eval_dataloader = DataLoader(eval_dataset, batch_size=4, shuffle=False, collate_fn=collate_fn)

    optimizer = optim.AdamW(model_wrapper.model.parameters(), lr=5e-5)

    logging.info("Starting training...")
    for epoch in range(3):
        model_wrapper.model.train()
        epoch_loss = 0
        for batch in train_dataloader:
            inputs = batch['input_ids']
            labels = batch['labels']

            optimizer.zero_grad()
            outputs = model_wrapper.model(inputs, inputs)
            loss = CrossEntropyLoss()(outputs.view(-1, model_wrapper.vocab_size), labels.view(-1))
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
            logging.info(f"Batch Loss: {loss.item()}")

        logging.info(f"Epoch: {epoch + 1}, Loss: {epoch_loss / len(train_dataloader)}")

    logging.info("Training completed.")

    output_dir = os.path.abspath(os.path.join(os.getcwd(), 'models'))
    os.makedirs(output_dir, exist_ok=True)

    model_wrapper.save(os.path.join(output_dir, 'aarambh_model.pth'), optimizer, 3)
    tokenizer.save_vocab('aarambh_vocab.json')
    logging.info(f"Model and tokenizer saved to {output_dir}.")

if __name__ == "__main__":
    script_dir = os.getcwd()
    dataset_path = os.path.join(script_dir, 'data/preprocessed_data.json')

    logging.info(f"Loading dataset from {dataset_path}...")
    with open(dataset_path, 'r') as f:
        articles = json.load(f)
    dataframe = pd.DataFrame(articles)
    
    # Print column names for debugging
    logging.info(f"Dataframe columns: {dataframe.columns}")
    
    logging.info("Dataset loaded.")
    train_model(dataframe)
