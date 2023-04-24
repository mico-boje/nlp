import os
import re
import json
import evaluate
import numpy as np

from datasets import Dataset, load_dataset
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, Seq2SeqTrainingArguments, Seq2SeqTrainer
from transformers import DataCollatorForSeq2Seq

from nlp.utils.utility import get_root_path

CHECKPOINT = "t5-small"
PREFIX = "summarize: "
TOKENIZER = AutoTokenizer.from_pretrained(CHECKPOINT, 
                                          use_fast=False, 
                                          model_max_length=1024, 
                                          truncation_side="left",
                                          )

def main():
    dataset = load_dataset('wikiann', 'en')
    tokenized_dataset = dataset.map(
                                preprocess_function, 
                                batched=False,
                                remove_columns=dataset["train"].column_names,)
    data_collator = DataCollatorForSeq2Seq(tokenizer=TOKENIZER, model=CHECKPOINT)
    model = AutoModelForSeq2SeqLM.from_pretrained(CHECKPOINT)
    # Define training arguments
    training_args = Seq2SeqTrainingArguments(
        output_dir=os.path.join(get_root_path(), 'data', 'models', 'checkpoint'),
        evaluation_strategy="epoch",
        learning_rate=1e-5,
        per_device_train_batch_size=4,
        per_device_eval_batch_size=4,
        weight_decay=0.01,
        save_total_limit=3,
        num_train_epochs=24,
        predict_with_generate=True,
        fp16=False,
        push_to_hub=False,
    )
    # Train model
    trainer = Seq2SeqTrainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_dataset["validation"],
        eval_dataset=tokenized_dataset["test"],
        tokenizer=TOKENIZER,
        data_collator=data_collator,
    )
    trainer.train()
    trainer.save_model(output_dir=os.path.join(get_root_path(), 'data', 'models', CHECKPOINT))
    

def preprocess_function(input):
    text = ' '.join(input["tokens"])
    # split spans into tags. Keep only the tag from before and only one instance of each tag: 
    # ['ORG: R.H. Saunders', 'ORG: St. Lawrence River'] -> ['ORG']
    tags = set([span.split(':')[0] for span in input["spans"]])
    #convert tags to string
    tags_str = ' '.join(tags)
    input_text = f'tags: {tags_str} text: {text}'
    model_inputs = TOKENIZER(input_text, max_length=1024, truncation=True, padding=True)
    target = ' '.join(input["spans"])
    labels = TOKENIZER(text_target=target, max_length=128, truncation=True, padding=True)
    model_inputs["labels"] = labels["input_ids"]
    return model_inputs
    
if __name__ == "__main__":
    main()