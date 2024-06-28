import os
import sys
import torch
from torch.utils.data import DataLoader

# Add the src directory to the system path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), 'rc')))

from aarambh.aarambh_wrapper import AarambhWrapper
from data.dataset import QADataset
from aarambh.aarambh_tokenizer import AarambhTokenizer

# Sample validation data
questions = ["Who developed you?", "What are you used for?"]
contexts = ["I was developed by a team of AI enthusiasts.", "I am used for natural language processing tasks such as question-answering."]
answers = ["a team of AI enthusiasts", "natural language processing tasks"]

# Initialize tokenizer and build vocabulary
tokenizer = AarambhTokenizer()
tokenizer.build_vocab(questions + contexts + answers)

# Initialize dataset and dataloader
dataset = QADataset(questions, contexts, answers, tokenizer)

def collate_fn(batch):
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

dataloader = DataLoader(dataset, batch_size=2, shuffle=False, collate_fn=collate_fn)

# Initialize model
model_wrapper = AarambhWrapper(
    vocab_size=100,
    d_model=512,
    nhead=8,
    num_encoder_layers=6,
    num_decoder_layers=6,
    dim_feedforward=2048,
    max_seq_length=5000
)

# Load the trained model
optimizer = torch.optim.Adam(model_wrapper.model.parameters())
model_wrapper.load('models/aarambh_model.pth', optimizer)

# Evaluation loop
def evaluate(model_wrapper, dataloader):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model_wrapper.model.to(device)
    model.eval()
    
    total_loss = 0
    with torch.no_grad():
        for batch in dataloader:
            question_ids = batch['question_ids'].to(device)
            context_ids = batch['context_ids'].to(device)
            answer_ids = batch['answer_ids'].to(device)

            # Ensure the source and target sequences are of the same length
            src_ids = torch.cat([question_ids, context_ids], dim=1)
            tgt_ids = answer_ids

            max_len_src = max(src_ids.shape[1], tgt_ids.shape[1])
            src_ids_padded = torch.cat([src_ids, torch.zeros(src_ids.shape[0], max_len_src - src_ids.shape[1]).long().to(device)], dim=1)
            tgt_ids_padded = torch.cat([tgt_ids, torch.zeros(tgt_ids.shape[0], max_len_src - tgt_ids.shape[1]).long().to(device)], dim=1)

            outputs = model(src_ids_padded, tgt_ids_padded)
            loss = torch.nn.CrossEntropyLoss()(outputs.view(-1, model_wrapper.vocab_size), tgt_ids_padded.view(-1))
            total_loss += loss.item()

    print(f"Validation Loss: {total_loss / len(dataloader)}")

# Evaluate the model
evaluate(model_wrapper, dataloader)