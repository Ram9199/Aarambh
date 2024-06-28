import torch
from torch.utils.data import Dataset

class QADataset(Dataset):
    def __init__(self, questions, contexts, answers, tokenizer):
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

        question_ids = torch.tensor(self.tokenizer.tokenize(question), dtype=torch.long)
        context_ids = torch.tensor(self.tokenizer.tokenize(context), dtype=torch.long)
        answer_ids = torch.tensor(self.tokenizer.tokenize(answer), dtype=torch.long)

        return {
            'question_ids': question_ids,
            'context_ids': context_ids,
            'answer_ids': answer_ids
        }
