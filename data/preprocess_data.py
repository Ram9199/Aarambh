import json
import os

def preprocess_jsonl(file_path):
    questions, contexts, answers = [], [], []
    with open(file_path, 'r', encoding='utf-8') as file:
        for line in file:
            item = json.loads(line)
            print(f"Processing item: {item}")
            questions.append(item.get('title', ''))
            contexts.append(item.get('paragraph', ''))
            answers.append(item.get('paragraph', ''))
    return questions, contexts, answers

def preprocess_data(data_dir, files):
    all_questions, all_contexts, all_answers = [], [], []
    
    for file in files:
        file_path = os.path.join(data_dir, file)
        questions, contexts, answers = preprocess_jsonl(file_path)
        all_questions.extend(questions)
        all_contexts.extend(contexts)
        all_answers.extend(answers)
    
    return all_questions, all_contexts, all_answers

def load_existing_data(file_path):
    if os.path.exists(file_path):
        with open(file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        return data['questions'], data['contexts'], data['answers']
    return [], [], []

def save_preprocessed_data(file_path, questions, contexts, answers):
    preprocessed_data = {
        "questions": questions,
        "contexts": contexts,
        "answers": answers
    }
    with open(file_path, 'w', encoding='utf-8') as f:
        json.dump(preprocessed_data, f, indent=4)

def remove_duplicates(questions, contexts, answers):
    unique_data = set()
    unique_questions, unique_contexts, unique_answers = [], [], []

    for q, c, a in zip(questions, contexts, answers):
        if (q, c, a) not in unique_data:
            unique_data.add((q, c, a))
            unique_questions.append(q)
            unique_contexts.append(c)
            unique_answers.append(a)

    return unique_questions, unique_contexts, unique_answers

if __name__ == "__main__":
    data_dir = os.path.abspath('data')
    files = ['simple_wikipedia_wikipedia_articles.jsonl', 'wikipedia_wikipedia_articles.jsonl']

    questions, contexts, answers = preprocess_data(data_dir, files)

    preprocessed_data_path = os.path.join(data_dir, 'preprocessed_data.json')
    existing_questions, existing_contexts, existing_answers = load_existing_data(preprocessed_data_path)

    all_questions = existing_questions + questions 
    all_contexts = existing_contexts + contexts
    all_answers = existing_answers + answers

    unique_questions, unique_contexts, unique_answers = remove_duplicates(all_questions, all_contexts, all_answers)

    save_preprocessed_data(preprocessed_data_path, unique_questions, unique_contexts, unique_answers)

    print("Preprocessing completed and data saved.")
    print(f"Total questions: {len(unique_questions)}")
    print(f"Total contexts: {len(unique_contexts)}")
    print(f"Total answers: {len(unique_answers)}")
