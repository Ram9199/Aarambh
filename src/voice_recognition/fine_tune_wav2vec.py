import torch
import torchaudio
from transformers import Wav2Vec2ForCTC, Wav2Vec2Processor, TrainingArguments, Trainer
from datasets import load_dataset, load_metric

# Load dataset
dataset = load_dataset("common_voice", "en", split="train+validation")

# Preprocess dataset
processor = Wav2Vec2Processor.from_pretrained("facebook/wav2vec2-base-960h")

def preprocess_function(examples):
    audio = examples["audio"]
    examples["input_values"] = processor(audio["array"], sampling_rate=audio["sampling_rate"]).input_values[0]
    examples["labels"] = processor.tokenizer(examples["sentence"]).input_ids
    return examples

dataset = dataset.map(preprocess_function, remove_columns=["audio"])

# Define data collator
data_collator = DataCollatorCTCWithPadding(processor=processor)

# Load pretrained model
model = Wav2Vec2ForCTC.from_pretrained("facebook/wav2vec2-base-960h")

# Define training arguments
training_args = TrainingArguments(
    output_dir="./wav2vec2",
    evaluation_strategy="steps",
    per_device_train_batch_size=8,
    per_device_eval_batch_size=8,
    num_train_epochs=3,
    save_steps=500,
    save_total_limit=2,
    learning_rate=2e-5,
    warmup_steps=500,
)

# Define metric
wer_metric = load_metric("wer")

def compute_metrics(pred):
    pred_ids = pred.predictions.argmax(-1)
    pred.label_ids[pred.label_ids == -100] = processor.tokenizer.pad_token_id
    pred_str = processor.batch_decode(pred_ids)
    label_str = processor.batch_decode(pred.label_ids, group_tokens=False)
    wer = wer_metric.compute(predictions=pred_str, references=label_str)
    return {"wer": wer}

# Define trainer
trainer = Trainer(
    model=model,
    args=training_args,
    data_collator=data_collator,
    compute_metrics=compute_metrics,
    train_dataset=dataset["train"],
    eval_dataset=dataset["validation"],
)

# Train model
trainer.train()

# Save model
model.save_pretrained("./wav2vec2")
processor.save_pretrained("./wav2vec2")
