import torch
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.nn.utils.rnn import pad_sequence
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'src')))
from aarambh.aarambh_wrapper import AarambhWrapper
from aarambh.aarambh_tokenizer import AarambhTokenizer

# Define paths
ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
MODEL_DIR = os.path.join(ROOT_DIR, 'models')
VOCAB_PATH = os.path.join(MODEL_DIR, 'aarambh_vocab.json')
MODEL_PATH = os.path.join(MODEL_DIR, 'aarambh_model.pth')
INCREMENTAL_MODEL_PATH = os.path.join(MODEL_DIR, 'aarambh_model_incremental.pth')

class CustomDataset(torch.utils.data.Dataset):
    def __init__(self, questions, contexts, answers, tokenizer):
        assert len(questions) == len(contexts), "The number of questions and contexts must be the same"
        self.questions = questions
        self.contexts = contexts
        self.answers = answers
        self.tokenizer = tokenizer

    def __len__(self):
        return len(self.questions)

    def __getitem__(self, idx):
        question = self.questions[idx]
        context = self.contexts[idx]
        answer = self.answers[idx]
        return {
            'question_ids': self.tokenizer.tokenize(question),
            'context_ids': self.tokenizer.tokenize(context),
            'answer_ids': self.tokenizer.tokenize(answer)
        }

def pad_collate(batch):
    question_ids = pad_sequence([torch.tensor(item['question_ids']) for item in batch], batch_first=True, padding_value=0)
    context_ids = pad_sequence([torch.tensor(item['context_ids']) for item in batch], batch_first=True, padding_value=0)
    answer_ids = pad_sequence([torch.tensor(item['answer_ids']) for item in batch], batch_first=True, padding_value=0)
    
    # Ensure the sequence lengths and batch sizes match
    combined_ids = torch.cat((question_ids, context_ids), dim=1)
    return {'combined_ids': combined_ids, 'answer_ids': answer_ids}

def create_dataloader(questions, contexts, answers, tokenizer, batch_size=2):
    dataset = CustomDataset(questions, contexts, answers, tokenizer)
    return DataLoader(dataset, batch_size=batch_size, shuffle=True, collate_fn=pad_collate)

def train_incrementally(model_wrapper, dataloaders, optimizer, epochs=3, start_epoch=0):
    model_wrapper.model.train()
    
    for epoch in range(start_epoch, start_epoch + epochs):
        epoch_loss = 0
        for dataloader in dataloaders:
            for batch in dataloader:
                combined_ids = batch['combined_ids']
                answer_ids = batch['answer_ids']

                optimizer.zero_grad()
                outputs = model_wrapper.model(combined_ids, combined_ids)
                
                # Reshape outputs to match the batch size of answer_ids
                outputs = outputs.view(-1, model_wrapper.vocab_size)
                answer_ids = answer_ids.view(-1)
                
                # Ensure the batch sizes match before calculating loss
                if outputs.size(0) != answer_ids.size(0):
                    min_size = min(outputs.size(0), answer_ids.size(0))
                    outputs = outputs[:min_size]
                    answer_ids = answer_ids[:min_size]

                loss = torch.nn.CrossEntropyLoss()(outputs, answer_ids)
                loss.backward()
                optimizer.step()
                epoch_loss += loss.item()
                print(f"Batch Loss: {loss.item()}")
        print(f"Epoch: {epoch + 1}, Loss: {epoch_loss / sum(len(dl) for dl in dataloaders)}")

def main():
    questions_list = [
        ["New question set 1?", "New question set 2?"],
        ["Another question set 1?", "Another question set 2?"]
    ]
    contexts_list = [
        ["New context set 1.", "New context set 2."],
        ["Another context set 1.", "Another context set 2."]
    ]
    answers_list = [
        ["New answer set 1", "New answer set 2"],
        ["Another answer set 1", "Another answer set 2"]
    ]

    tokenizer = AarambhTokenizer()
    print(f"Loading vocab from: {VOCAB_PATH}")
    tokenizer.load_vocab(VOCAB_PATH)
    vocab_size = tokenizer.vocab_size  # Ensure vocab_size is loaded correctly

    dataloaders = [create_dataloader(questions, contexts, answers, tokenizer) for questions, contexts, answers in zip(questions_list, contexts_list, answers_list)]

    # Ensure max_seq_length is consistent with the saved model
    max_seq_length = 5000  # This should match the value used when the model was saved

    model_wrapper = AarambhWrapper(
        vocab_size=vocab_size,
        d_model=512,
        nhead=8,
        num_encoder_layers=6,
        num_decoder_layers=6,
        dim_feedforward=2048,
        max_seq_length=max_seq_length
    )

    optimizer = optim.Adam(model_wrapper.model.parameters(), lr=5e-5)
    
    # Load the model from the root directory
    start_epoch = model_wrapper.load(MODEL_PATH, optimizer)

    train_incrementally(model_wrapper, dataloaders, optimizer, epochs=3, start_epoch=start_epoch)

    # Save the model to the root directory
    model_wrapper.save(INCREMENTAL_MODEL_PATH, optimizer, start_epoch + 3)

if __name__ == "__main__":
    main()
