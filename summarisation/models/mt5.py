import os
import json
import evaluate
import numpy as np

from datasets import Dataset
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, Seq2SeqTrainingArguments, Seq2SeqTrainer
from transformers import DataCollatorForSeq2Seq

from summarisation.utils.utility import get_root_path

CHECKPOINT = "google/mt5-small"
PREFIX = "summarize: "
TOKENIZER = AutoTokenizer.from_pretrained(CHECKPOINT, use_fast=False)
ROUGE = evaluate.load("rouge")

def main():
    # Load data
    dataset = create_train_test_split()
    tokenized_dataset = dataset.map(preprocess_function, batched=True)
    data_collator = DataCollatorForSeq2Seq(tokenizer=TOKENIZER, model=CHECKPOINT)
    # Load Model
    model = AutoModelForSeq2SeqLM.from_pretrained(CHECKPOINT)
    # Define training arguments
    training_args = Seq2SeqTrainingArguments(
        output_dir=os.path.join(get_root_path(), 'data', 'models', 'checkpoint'),
        evaluation_strategy="epoch",
        learning_rate=2e-5,
        per_device_train_batch_size=2,
        per_device_eval_batch_size=2,
        weight_decay=0.01,
        save_total_limit=3,
        num_train_epochs=4,
        predict_with_generate=True,
        fp16=True,
        push_to_hub=False,
    )
    # Train model
    trainer = Seq2SeqTrainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_dataset["train"],
        eval_dataset=tokenized_dataset["test"],
        tokenizer=TOKENIZER,
        data_collator=data_collator,
        compute_metrics=compute_metrics,
    )
    trainer.train()
    trainer.save_model(output_dir=os.path.join(get_root_path(), 'data', 'models', 'mt5-small'))
    
def compute_metrics(eval_pred):
    predictions, labels = eval_pred
    decoded_preds = TOKENIZER.batch_decode(predictions, skip_special_tokens=True)
    labels = np.where(labels != -100, labels, TOKENIZER.pad_token_id)
    decoded_labels = TOKENIZER.batch_decode(labels, skip_special_tokens=True)
    result = ROUGE.compute(predictions=decoded_preds, references=decoded_labels, use_stemmer=True)
    prediction_lens = [np.count_nonzero(pred != TOKENIZER.pad_token_id) for pred in predictions]
    result["gen_len"] = np.mean(prediction_lens)

    return {k: round(v, 4) for k, v in result.items()}

def create_train_test_split(train_size=0.7):
    text = []
    summary = []
    data_path = os.path.join(get_root_path(), 'data', 'json')
    
    for pdf in os.listdir(data_path):
        with open(os.path.join(data_path, pdf), 'r') as f:
            data = json.load(f)
            text.append(data['text'])
            summary.append(data['summary'])
    dataset = Dataset.from_dict({"text": text, "summary": summary})
    return dataset.train_test_split(train_size=train_size)


def preprocess_function(examples):
    inputs = [PREFIX + doc for doc in examples["text"]]
    model_inputs = TOKENIZER(inputs, max_length=1024, truncation=True)
    labels = TOKENIZER(text_target=examples["summary"], max_length=128, truncation=True)
    model_inputs["labels"] = labels["input_ids"]
    return model_inputs
    
if __name__ == "__main__":
    main()
