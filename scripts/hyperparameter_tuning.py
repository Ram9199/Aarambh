import optuna
import torch
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.nn.utils.rnn import pad_sequence
import sys
import os
import logging
from sklearn.model_selection import train_test_split

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'src')))
from aarambh.aarambh_wrapper import AarambhWrapper
from aarambh.aarambh_tokenizer import AarambhTokenizer

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Define paths
ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
MODEL_DIR = os.path.join(ROOT_DIR, 'models')
VOCAB_PATH = os.path.join(MODEL_DIR, 'aarambh_vocab.json')

def objective(trial):
    # Define hyperparameters to tune
    d_model = trial.suggest_categorical('d_model', [256, 512])
    nhead = trial.suggest_categorical('nhead', [4, 8])
    num_encoder_layers = trial.suggest_int('num_encoder_layers', 2, 6)
    num_decoder_layers = trial.suggest_int('num_decoder_layers', 2, 6)
    dim_feedforward = trial.suggest_int('dim_feedforward', 512, 2048, step=512)
    lr = trial.suggest_loguniform('lr', 1e-5, 1e-3)

    # Define your data
    questions = ["What is AI?", "How does a neural network work?"]
    contexts = ["AI stands for Artificial Intelligence.", "A neural network is a series of algorithms."]
    answers = ["Artificial Intelligence", "series of algorithms"]

    # Split data into training and validation sets
    questions_train, questions_val, contexts_train, contexts_val, answers_train, answers_val = train_test_split(questions, contexts, answers, test_size=0.2, random_state=42)

    # Initialize the model
    tokenizer = AarambhTokenizer()
    texts = questions + contexts + answers
    tokenizer.build_vocab(texts)
    model_wrapper = AarambhWrapper(
        vocab_size=tokenizer.vocab_size,
        d_model=d_model,
        nhead=nhead,
        num_encoder_layers=num_encoder_layers,
        num_decoder_layers=num_decoder_layers,
        dim_feedforward=dim_feedforward,
        max_seq_length=5000
    )

    # DataLoader
    def pad_collate(batch):
        question_ids = pad_sequence([torch.tensor(item['question_ids']) for item in batch], batch_first=True, padding_value=0)
        context_ids = pad_sequence([torch.tensor(item['context_ids']) for item in batch], batch_first=True, padding_value=0)
        answer_ids = pad_sequence([torch.tensor(item['answer_ids']) for item in batch], batch_first=True, padding_value=0)

        # Ensure the batch sizes are the same
        min_batch_size = min(question_ids.size(0), context_ids.size(0), answer_ids.size(0))
        question_ids = question_ids[:min_batch_size]
        context_ids = context_ids[:min_batch_size]
        answer_ids = answer_ids[:min_batch_size]

        # Return a single tensor to be used for both src and tgt
        combined_ids = torch.cat((question_ids, context_ids), dim=1)

        return {'combined_ids': combined_ids, 'answer_ids': answer_ids}

    def create_dataloader(questions, contexts, answers, tokenizer, batch_size):
        dataset = CustomDataset(questions, contexts, answers, tokenizer)
        return DataLoader(dataset, batch_size=batch_size, shuffle=True, collate_fn=pad_collate)

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

    train_loader = create_dataloader(questions_train, contexts_train, answers_train, tokenizer, batch_size=4)
    val_loader = create_dataloader(questions_val, contexts_val, answers_val, tokenizer, batch_size=4)

    # Training
    model_wrapper.model.train()
    optimizer = optim.Adam(model_wrapper.model.parameters(), lr=lr)
    criterion = torch.nn.CrossEntropyLoss()

    for epoch in range(3):
        for batch in train_loader:
            optimizer.zero_grad()
            combined_ids = batch['combined_ids']
            answer_ids = batch['answer_ids']

            # Ensure that the sequence lengths and batch sizes match
            combined_ids = combined_ids.view(-1, combined_ids.size(-1))
            answer_ids = answer_ids.view(-1)

            outputs = model_wrapper.model(combined_ids, combined_ids)
            outputs = outputs.view(-1, model_wrapper.vocab_size)

            # Ensure outputs and answer_ids have the same batch size
            if outputs.size(0) != answer_ids.size(0):
                min_size = min(outputs.size(0), answer_ids.size(0))
                outputs = outputs[:min_size]
                answer_ids = answer_ids[:min_size]

            loss = criterion(outputs, answer_ids)
            loss.backward()
            optimizer.step()

    # Validation
    model_wrapper.model.eval()
    val_loss = 0
    with torch.no_grad():
        for batch in val_loader:
            combined_ids = batch['combined_ids']
            answer_ids = batch['answer_ids']

            # Ensure that the sequence lengths and batch sizes match
            combined_ids = combined_ids.view(-1, combined_ids.size(-1))
            answer_ids = answer_ids.view(-1)

            outputs = model_wrapper.model(combined_ids, combined_ids)
            outputs = outputs.view(-1, model_wrapper.vocab_size)

            # Ensure outputs and answer_ids have the same batch size
            if outputs.size(0) != answer_ids.size(0):
                min_size = min(outputs.size(0), answer_ids.size(0))
                outputs = outputs[:min_size]
                answer_ids = answer_ids[:min_size]

            loss = criterion(outputs, answer_ids)
            val_loss += loss.item()

    avg_val_loss = val_loss / len(val_loader)
    return avg_val_loss

# Optuna study
study = optuna.create_study(direction='minimize')
study.optimize(objective, n_trials=20)

# Best hyperparameters
print("Best hyperparameters: ", study.best_params)

# Save best model
best_trial = study.best_trial
best_params = best_trial.params

# Initialize the model with the best hyperparameters
tokenizer = AarambhTokenizer()
tokenizer.load_vocab(VOCAB_PATH)
model_wrapper = AarambhWrapper(
    vocab_size=tokenizer.vocab_size,
    d_model=best_params['d_model'],
    nhead=best_params['nhead'],
    num_encoder_layers=best_params['num_encoder_layers'],
    num_decoder_layers=best_params['num_decoder_layers'],
    dim_feedforward=best_params['dim_feedforward'],
    max_seq_length=5000
)

# Save the best model
model_wrapper.save(os.path.join(MODEL_DIR, 'aarambh_best_model.pth'), optim.Adam(model_wrapper.model.parameters()), 1)
