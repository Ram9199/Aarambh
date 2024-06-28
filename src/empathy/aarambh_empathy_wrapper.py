import torch
import torch.optim as optim

from .aarambh_empathy import AarambhEmpathy
from .aarambh_empathy_tokenizer import AarambhTokenizer

class AarambhEmpathyWrapper:
    def __init__(self, vocab_size=50257, d_model=512, nhead=8, num_encoder_layers=6, num_decoder_layers=6, dim_feedforward=2048, max_seq_length=512):
        self.tokenizer = AarambhTokenizer()
        self.model = AarambhEmpathy(vocab_size, d_model, nhead, num_encoder_layers, num_decoder_layers, dim_feedforward, max_seq_length)

    def generate_empathetic_response(self, prompt, max_length=50):
        self.model.eval()
        with torch.no_grad():
            tokens = self.tokenizer.tokenize(prompt)
            input_ids = torch.tensor([tokens])
            outputs = self.model(input_ids, input_ids)
            response_tokens = outputs.argmax(dim=-1).squeeze().tolist()
            response = self.tokenizer.detokenize(response_tokens)
            return response

    def train(self, dataset, epochs=3, batch_size=2, lr=5e-5):
        self.model.train()
        optimizer = optim.AdamW(self.model.parameters(), lr=lr)

        for epoch in range(epochs):
            for batch in dataset.batch(batch_size):
                input_texts = [text for text in batch]
                inputs = [self.tokenizer.tokenize(text) for text in input_texts]
                inputs = torch.nn.utils.rnn.pad_sequence([torch.tensor(x) for x in inputs], batch_first=True, padding_value=self.tokenizer.vocab["<pad>"])
                labels = inputs.clone()
                optimizer.zero_grad()
                outputs = self.model(inputs, inputs)
                loss = torch.nn.CrossEntropyLoss()(outputs.view(-1, outputs.size(-1)), labels.view(-1))
                loss.backward()
                optimizer.step()
                print(f"Epoch: {epoch}, Loss: {loss.item()}")

    def save(self, model_path, vocab_path):
        # Save the model's state dictionary with the correct positional encoder shape
        pos_encoder_state_dict = self.model.pos_encoder.state_dict()
        for key, value in pos_encoder_state_dict.items():
            if "pos_encoder.embedding.weight" in key:
                pos_encoder_state_dict[key] = value.view(1, -1, value.size(-1))
        self.model.pos_encoder.load_state_dict(pos_encoder_state_dict)
        torch.save(self.model.state_dict(), model_path)
        self.tokenizer.save_vocab(vocab_path)

    def load(self, model_path, vocab_path):
        # Load the model's state dictionary with the correct positional encoder shape
        self.model.load_state_dict(torch.load(model_path))
        self.tokenizer.load_vocab(vocab_path)
        pos_encoder_state_dict = self.model.pos_encoder.state_dict()
        for key, value in pos_encoder_state_dict.items():
            if "pos_encoder.embedding.weight" in key:
                pos_encoder_state_dict[key] = value.view(1, -1, value.size(-1))
        self.model.pos_encoder.load_state_dict(pos_encoder_state_dict)