import torch
import torch.optim as optim
from torch.utils.data import DataLoader

from .aarambh import Aarambh
from .aarambh_tokenizer import AarambhTokenizer

class AarambhWrapper:
    def __init__(self, vocab_size=50257, d_model=512, nhead=8, num_encoder_layers=6, num_decoder_layers=6, dim_feedforward=2048, max_seq_length=512):
        self.tokenizer = AarambhTokenizer()
        self.model = Aarambh(vocab_size, d_model, nhead, num_encoder_layers, num_decoder_layers, dim_feedforward, max_seq_length)

    def generate_response(self, prompt, max_length=50):
        tokens = self.tokenizer.tokenize(prompt)
        print("Tokenized input:", tokens)
        input_ids = torch.tensor([tokens])
        self.model.eval()
        with torch.no_grad():
            outputs = self.model(input_ids, input_ids)
            response_tokens = outputs.argmax(dim=-1).squeeze().tolist()
            print("Response tokens:", response_tokens)
            response = self.tokenizer.detokenize(response_tokens)
            return response

    def train(self, dataset, epochs=3, batch_size=2, lr=5e-5):
        self.model.train()
        optimizer = optim.AdamW(self.model.parameters(), lr=lr)
        data_loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
        
        for epoch in range(epochs):
            epoch_loss = 0
            for batch in data_loader:
                inputs = [self.tokenizer.tokenize(text) for text in batch]
                inputs = torch.nn.utils.rnn.pad_sequence([torch.tensor(x) for x in inputs], batch_first=True, padding_value=self.tokenizer.vocab["<pad>"])
                labels = inputs.clone()
                optimizer.zero_grad()
                outputs = self.model(inputs, inputs)
                loss = torch.nn.CrossEntropyLoss()(outputs.view(-1, outputs.size(-1)), labels.view(-1))
                loss.backward()
                optimizer.step()
                epoch_loss += loss.item()
                print(f"Batch Loss: {loss.item()}")
            print(f"Epoch: {epoch}, Loss: {epoch_loss / len(data_loader)}")

    def save(self, model_path, vocab_path):
        torch.save(self.model.state_dict(), model_path)
        self.tokenizer.save_vocab(vocab_path)

    def load(self, model_path, vocab_path):
        self.model.load_state_dict(torch.load(model_path))
        self.tokenizer.load_vocab(vocab_path)
