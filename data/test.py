import json
import os

def preprocess_jsonl(file_path):
    questions = []
    contexts = []
    answers = []
    
    with open(file_path, 'r', encoding='utf-8') as file:
        for line in file:
            data = json.loads(line.strip())
            question = data.get('title', "")
            context = data.get('paragraph', "")
            answer = data.get('paragraph', "")
            
            if question and context:
                questions.append(question)
                contexts.append(context)
                answers.append(answer)
    
    return questions, contexts, answers

def save_preprocessed_data(file_path, questions, contexts, answers):
    preprocessed_data = {
        "questions": questions,
        "contexts": contexts,
        "answers": answers
    }
    
    with open(file_path, 'w', encoding='utf-8') as f:
        json.dump(preprocessed_data, f, indent=4)

# Paths to the input and output files
data_dir = os.path.abspath('.')
input_files = ['simple_wikipedia_wikipedia_articles.jsonl', 'wikipedia_wikipedia_articles.jsonl']
output_file = 'preprocessed_data.json'

all_questions = []
all_contexts = []
all_answers = []

for input_file in input_files:
    questions, contexts, answers = preprocess_jsonl(input_file)
    all_questions.extend(questions)
    all_contexts.extend(contexts)
    all_answers.extend(answers)

save_preprocessed_data(output_file, all_questions, all_contexts, all_answers)

print(f"Preprocessed data saved to {output_file}")
