import json
import os
from aarambh.aarambh_tokenizer import AarambhTokenizer

def preprocess_data(input_filepath, output_filepath, tokenizer):
    with open(input_filepath, 'r') as f:
        data = [json.loads(line) for line in f]

    tokenized_data = []

    for item in data:
        question_tokens = tokenizer.tokenize(item['question'])
        answer_tokens = tokenizer.tokenize(item['answer'])
        tokenized_data.append({"question_tokens": question_tokens, "answer_tokens": answer_tokens})

    with open(output_filepath, 'w') as f:
        for entry in tokenized_data:
            f.write(json.dumps(entry) + "\n")

def main():
    input_filepath = "data/raw/k12_questions_answers.jsonl"
    output_filepath = "data/processed/tokenized_k12_questions_answers.jsonl"
    
    tokenizer = AarambhTokenizer()
    
    # Load or build vocabulary
    if os.path.exists('models/vocab.json'):
        tokenizer.load_vocab('models/vocab.json')
    else:
        with open(input_filepath, 'r') as f:
            texts = [json.loads(line)['question'] + ' ' + json.loads(line)['answer'] for line in f]
        tokenizer.build_vocab(texts)
        tokenizer.save_vocab('models/vocab.json')
    
    preprocess_data(input_filepath, output_filepath, tokenizer)
    print(f"Preprocessed data saved to {output_filepath}")

if __name__ == "__main__":
    main()
