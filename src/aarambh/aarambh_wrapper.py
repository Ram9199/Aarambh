import torch
import torch.optim as optim
from torch.utils.data import DataLoader

from .aarambh import Aarambh
from .aarambh_tokenizer import AarambhTokenizer
from torch.nn.utils.rnn import pad_sequence

class AarambhWrapper:
    def __init__(self, vocab_size, d_model=512, nhead=8, num_encoder_layers=6, num_decoder_layers=6, dim_feedforward=2048, max_seq_length=512):
        self.vocab_size = vocab_size
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
        data_loader = DataLoader(dataset, batch_size=batch_size, shuffle=True, collate_fn=self.collate_fn)
        
        for epoch in range(epochs):
            epoch_loss = 0
            for batch in data_loader:
                question_ids = batch['question_ids']
                context_ids = batch['context_ids']
                answer_ids = batch['answer_ids']

                inputs = torch.cat((question_ids, context_ids), dim=1)
                labels = answer_ids.clone()

                optimizer.zero_grad()
                outputs = self.model(inputs, inputs)
                loss = torch.nn.CrossEntropyLoss()(outputs.view(-1, self.vocab_size), labels.view(-1))
                loss.backward()
                optimizer.step()
                epoch_loss += loss.item()
                print(f"Batch Loss: {loss.item()}")
            print(f"Epoch: {epoch + 1}, Loss: {epoch_loss / len(data_loader)}")

    def collate_fn(self, batch):
        max_len_question = max(len(item['question_ids']) for item in batch)
        max_len_context = max(len(item['context_ids']) for item in batch)
        max_len_answer = max(len(item['answer_ids']) for item in batch)
        
        max_len_src = max(max_len_question, max_len_context)
        max_len_tgt = max_len_answer

        padded_questions = []
        padded_contexts = []
        padded_answers = []

        for item in batch:
            question = item['question_ids']
            context = item['context_ids']
            answer = item['answer_ids']

            padded_question = torch.cat([question, torch.tensor([0] * (max_len_src - len(question)), dtype=torch.long)])
            padded_context = torch.cat([context, torch.tensor([0] * (max_len_src - len(context)), dtype=torch.long)])
            padded_answer = torch.cat([answer, torch.tensor([0] * (max_len_tgt - len(answer)), dtype=torch.long)])

            padded_questions.append(padded_question)
            padded_contexts.append(padded_context)
            padded_answers.append(padded_answer)

        return {
            'question_ids': torch.stack(padded_questions),
            'context_ids': torch.stack(padded_contexts),
            'answer_ids': torch.stack(padded_answers)
        }

    def save(self, model_path, optimizer, epoch):
        self.model.save(model_path, optimizer, epoch)
        self.tokenizer.save_vocab('models/aarambh_vocab.json')

    def load(self, model_path, optimizer=None):
        epoch = self.model.load(model_path, optimizer)
        self.tokenizer.load_vocab('models/aarambh_vocab.json')
        return epoch
