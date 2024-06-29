import json
import os

def preprocess_jsonl(file_path):
    questions, contexts, answers = [], [], []
    with open(file_path, 'r', encoding='utf-8') as file:
        for line in file:
            item = json.loads(line)
            # Debugging: Print the item to see its structure
            print(f"Processing item: {item}")
            # Assuming each item has 'title', 'paragraph', and 'paragraph' fields.
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

if __name__ == "__main__":
    # Ensure data_dir is an absolute path
    data_dir = os.path.abspath('data')
    files = ['simple_wikipedia_wikipedia_articles.jsonl', 'wikipedia_wikipedia_articles.jsonl']

    questions, contexts, answers = preprocess_data(data_dir, files)

    preprocessed_data_path = os.path.join(data_dir, 'preprocessed_data.json')
    with open(preprocessed_data_path, 'w', encoding='utf-8') as f:
        json.dump({
            'questions': questions,
            'contexts': contexts,
            'answers': answers
        }, f, indent=4)

    print("Preprocessing completed and data saved.")
    print(f"Total questions: {len(questions)}")
    print(f"Total contexts: {len(contexts)}")
    print(f"Total answers: {len(answers)}")