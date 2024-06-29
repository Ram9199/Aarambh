import os
import torch
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.nn.utils.rnn import pad_sequence
from sklearn.metrics import accuracy_score
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'src')))
from aarambh.aarambh_wrapper import AarambhWrapper
from aarambh.aarambh_tokenizer import AarambhTokenizer

# Define paths
ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
MODEL_DIR = os.path.join(ROOT_DIR, 'models')
VOCAB_PATH = os.path.join(MODEL_DIR, 'aarambh_vocab.json')
MODEL_PATHS = [os.path.join(MODEL_DIR, f'aarambh_model_{i}.pth') for i in range(3)]
INCREMENTAL_MODEL_PATHS = [os.path.join(MODEL_DIR, f'aarambh_model_incremental_{i}.pth') for i in range(3)]

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

    combined_ids = torch.cat((question_ids, context_ids), dim=1)

    return {'combined_ids': combined_ids, 'answer_ids': answer_ids}

def create_dataloader(questions, contexts, answers, tokenizer, batch_size=2):
    dataset = CustomDataset(questions, contexts, answers, tokenizer)
    return DataLoader(dataset, batch_size=batch_size, shuffle=True, collate_fn=pad_collate)

def evaluate(models, dataloader):
    predictions, true_labels = [], []
    
    for model_wrapper in models:
        model_wrapper.model.eval()

    with torch.no_grad():
        for batch in dataloader:
            combined_ids = batch['combined_ids']
            answer_ids = batch['answer_ids']

            outputs_list = [model_wrapper.model(combined_ids, combined_ids) for model_wrapper in models]
            outputs_avg = sum(outputs_list) / len(outputs_list)
            preds = outputs_avg.argmax(dim=-1).view(-1).tolist()
            true = answer_ids.view(-1).tolist()

            min_length = min(len(preds), len(true))
            predictions.extend(preds[:min_length])
            true_labels.extend(true[:min_length])
    
    accuracy = accuracy_score(true_labels, predictions)
    return accuracy

def train_incrementally(models, train_loader, val_loader, optimizers, epochs=3, start_epoch=0):
    for model_wrapper in models:
        model_wrapper.model.train()
    
    for epoch in range(start_epoch, start_epoch + epochs):
        epoch_loss = 0
        for model_wrapper, optimizer in zip(models, optimizers):
            for batch in train_loader:
                combined_ids = batch['combined_ids']
                answer_ids = batch['answer_ids']

                optimizer.zero_grad()
                outputs = model_wrapper.model(combined_ids, combined_ids)
                outputs = outputs.view(-1, model_wrapper.vocab_size)
                answer_ids = answer_ids.view(-1)

                # Ensure that the length of outputs and answer_ids match
                min_length = min(outputs.size(0), answer_ids.size(0))
                outputs = outputs[:min_length]
                answer_ids = answer_ids[:min_length]

                loss = torch.nn.CrossEntropyLoss()(outputs, answer_ids)
                loss.backward()
                optimizer.step()
                epoch_loss += loss.item()
                print(f"Batch Loss: {loss.item()}")
        
        avg_loss = epoch_loss / len(train_loader)
        print(f"Epoch: {epoch + 1}, Loss: {avg_loss}")
        
        # Evaluate on validation data
        val_accuracy = evaluate(models, val_loader)
        print(f"Validation Accuracy: {val_accuracy}")
        
        # Adjust learning rate based on validation accuracy
        if val_accuracy < 0.8:  # Example threshold
            for optimizer in optimizers:
                for param_group in optimizer.param_groups:
                    param_group['lr'] *= 0.9  # Decrease learning rate by 10%
                    print(f"Decreased learning rate to: {param_group['lr']}")

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
    
    train_loader = dataloaders[0]
    val_loader = dataloaders[1]

    max_seq_length = 5000  # This should match the value used when the model was saved

    models = []
    optimizers = []
    for model_path in MODEL_PATHS:
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
        start_epoch = model_wrapper.load(model_path, optimizer)
        models.append(model_wrapper)
        optimizers.append(optimizer)

    train_incrementally(models, train_loader, val_loader, optimizers, epochs=3, start_epoch=start_epoch)

    for i, model_wrapper in enumerate(models):
        model_wrapper.save(INCREMENTAL_MODEL_PATHS[i], optimizers[i], start_epoch + 3)

if __name__ == "__main__":
    main()
