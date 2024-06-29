import os
import sys
import torch
from torch.utils.data import DataLoader
import torch.optim as optim
from torch.nn import CrossEntropyLoss
import json

# Add the src directory to the system path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), 'src')))

from aarambh.aarambh_wrapper import AarambhWrapper
from aarambh.aarambh_tokenizer import AarambhTokenizer

def load_preprocessed_data(filepath):
    with open(filepath, 'r') as f:
        articles = json.load(f)
    return articles

def create_dataloader(articles, tokenizer, batch_size=2):
    dataset = WikipediaDataset(articles, tokenizer)
    return DataLoader(dataset, batch_size=batch_size, shuffle=True, collate_fn=collate_fn)

def collate_fn(batch):
    max_len = max(len(item['tokens']) for item in batch)

    padded_tokens = []
    for item in batch:
        tokens = item['tokens']
        padded = torch.cat([torch.tensor(tokens, dtype=torch.long), torch.zeros(max_len - len(tokens), dtype=torch.long)])
        padded_tokens.append(padded)

    return {'tokens': torch.stack(padded_tokens)}

class WikipediaDataset(torch.utils.data.Dataset):
    def __init__(self, articles, tokenizer):
        self.articles = articles
        self.tokenizer = tokenizer

    def __len__(self):
        return len(self.articles)

    def __getitem__(self, idx):
        article = self.articles[idx]
        return {'tokens': torch.tensor(article['tokens'])}

def train(model_wrapper, dataloader, epochs=3, lr=5e-5):
    model_wrapper.model.train()
    optimizer = optim.AdamW(model_wrapper.model.parameters(), lr=lr)

    for epoch in range(epochs):
        epoch_loss = 0
        for batch in dataloader:
            inputs = batch['tokens']
            labels = batch['tokens']

            optimizer.zero_grad()
            outputs = model_wrapper.model(inputs, inputs)
            loss = CrossEntropyLoss()(outputs.view(-1, model_wrapper.vocab_size), labels.view(-1))
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
            print(f"Batch Loss: {loss.item()}")

        print(f"Epoch: {epoch + 1}, Loss: {epoch_loss / len(dataloader)}")

    model_wrapper.save('models/aarambh_model.pth', optimizer, epochs)

def main():
    articles = load_preprocessed_data('data/raw/wikipedia_articles.json')

    tokenizer = AarambhTokenizer()
    texts = [article['content'] for article in articles]
    tokenizer.build_vocab(texts)

    tokenized_articles = [{'tokens': tokenizer.tokenize(article['content'])} for article in articles]
    dataloader = create_dataloader(tokenized_articles, tokenizer)

    model_wrapper = AarambhWrapper(
        vocab_size=tokenizer.vocab_size,
        d_model=512,
        nhead=8,
        num_encoder_layers=6,
        num_decoder_layers=6,
        dim_feedforward=2048,
        max_seq_length=512
    )

    os.makedirs('models', exist_ok=True)
    train(model_wrapper, dataloader, epochs=3, lr=5e-5)

if __name__ == "__main__":
    main()
