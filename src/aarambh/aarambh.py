import torch
import torch.nn as nn
import math
from src.data_loader import build_vocab

class Aarambh(nn.Module):
    def __init__(self, vocab_size, d_model, nhead, num_encoder_layers, num_decoder_layers, dim_feedforward, max_seq_length):
        super(Aarambh, self).__init__()
        self.d_model = d_model
        self.embedding = nn.Embedding(vocab_size, d_model)
        self.pos_encoder = PositionalEncoding(d_model, max_seq_length)
        self.transformer = nn.Transformer(
            d_model=d_model,
            nhead=nhead,
            num_encoder_layers=num_encoder_layers,
            num_decoder_layers=num_decoder_layers,
            dim_feedforward=dim_feedforward,
            batch_first=True  # Set batch_first to True
        )
        self.fc_out = nn.Linear(d_model, vocab_size)
        self._init_weights()

    def _init_weights(self):
        nn.init.kaiming_uniform_(self.embedding.weight, a=math.sqrt(5))

    def forward(self, src, tgt):
        if src.size(0) != tgt.size(0):
            raise ValueError("The batch size of src and tgt must be equal")

        src_len = src.size(1)
        tgt_len = tgt.size(1)

        src = self.embedding(src) * math.sqrt(self.d_model)
        src = self.pos_encoder(src, src_len)
        tgt = self.embedding(tgt) * math.sqrt(self.d_model)
        tgt = self.pos_encoder(tgt, tgt_len)

        output = self.transformer(src, tgt)
        output = self.fc_out(output)
        return output

    def save(self, model_path, optimizer, epoch, vocab_size):
        torch.save({
            'epoch': epoch,
            'model_state_dict': self.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'vocab_size': vocab_size,
            'pos_encoder': self.pos_encoder.pe
        }, model_path)

    def load(self, model_path, optimizer=None):
        checkpoint = torch.load(model_path)
        self.load_state_dict(checkpoint['model_state_dict'])
        self.pos_encoder.pe.copy_(checkpoint['pos_encoder'])
        if optimizer is not None:
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        return checkpoint['epoch'], checkpoint['vocab_size']

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=5000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=0.1)
        
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)

    def forward(self, x, length):
        x = x + self.pe[:, :length]
        return self.dropout(x)

# Example usage
preprocessed_data_path = r'D:\Aarambh\data\preprocessed_data.json'
vocab_path = r'D:\Aarambh\models\aarambh_vocab.json'
vocab = build_vocab(preprocessed_data_path, vocab_path)
vocab_size = len(vocab)

d_model = 512
nhead = 8
num_encoder_layers = 6
num_decoder_layers = 6
dim_feedforward = 2048
max_seq_length = 5000

model = Aarambh(vocab_size, d_model, nhead, num_encoder_layers, num_decoder_layers, dim_feedforward, max_seq_length)
optimizer = torch.optim.Adam(model.parameters())

# Save the model
model.save('models/aarambh_model.pth', optimizer, 1, vocab_size)

# Create a new optimizer object
new_optimizer = torch.optim.Adam(model.parameters())

# Load the model
loaded_epoch, loaded_vocab_size = model.load('models/aarambh_model.pth', new_optimizer)
print(f"Loaded model at epoch {loaded_epoch} with vocab size {loaded_vocab_size}")
