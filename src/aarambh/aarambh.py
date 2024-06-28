import torch
import torch.nn as nn
import math

class Aarambh(nn.Module):
    def __init__(self, vocab_size, d_model, nhead, num_encoder_layers, num_decoder_layers, dim_feedforward, max_seq_length):
        super(Aarambh, self).__init__()
        self.embedding = nn.Embedding(vocab_size, d_model)
        self.pos_encoder = PositionalEncoding(d_model, max_seq_length)
        self.transformer = nn.Transformer(d_model, nhead, num_encoder_layers, num_decoder_layers, dim_feedforward)
        self.fc_out = nn.Linear(d_model, vocab_size)
        self._init_weights()

    def _init_weights(self):
        nn.init.kaiming_uniform_(self.embedding.weight, a=math.sqrt(5))

    def forward(self, src, tgt):
        src_emb = self.embedding(src) + self.pos_encoder(src)
        tgt_emb = self.embedding(tgt) + self.pos_encoder(tgt)
        output = self.transformer(src_emb, tgt_emb)
        output = self.fc_out(output)
        return output

    def save(self, model_path, optimizer, epoch):
        torch.save({
            'epoch': epoch,
            'model_state_dict': self.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'pos_encoder': self.pos_encoder.pe
        }, model_path)

    def load(self, model_path, optimizer=None):
        checkpoint = torch.load(model_path)
        self.load_state_dict(checkpoint['model_state_dict'])
        self.pos_encoder.pe.copy_(checkpoint['pos_encoder'])
        if optimizer is not None:
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        return checkpoint['epoch']

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=5000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=0.1)
        
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        self.register_buffer('pe', pe)

    def forward(self, x):
        return self.pe[:x.size(0), :]

# Example usage
vocab_size = 50257
d_model = 512
nhead = 8
num_encoder_layers = 6
num_decoder_layers = 6
dim_feedforward = 2048
max_seq_length = 5000

model = Aarambh(vocab_size, d_model, nhead, num_encoder_layers, num_decoder_layers, dim_feedforward, max_seq_length)
optimizer = torch.optim.Adam(model.parameters())

# Save the model
model.save('models/aarambh_model.pth', optimizer, 1)

# Create a new optimizer object
new_optimizer = torch.optim.Adam(model.parameters())

# Load the model
loaded_epoch = model.load('models/aarambh_model.pth', new_optimizer)
print(f"Loaded model at epoch {loaded_epoch}")