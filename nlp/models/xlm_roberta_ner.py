import os
import evaluate
import numpy as np

from datasets import load_dataset, concatenate_datasets
from transformers import (AutoTokenizer, 
                          DataCollatorForTokenClassification, 
                          AutoModelForTokenClassification, 
                          TrainingArguments,
                          Trainer,)

from nlp.utils.utility import get_root_path

# Model
CHECKPOINT = 'xlm-roberta-base'
TOKENIZER = AutoTokenizer.from_pretrained(CHECKPOINT)

# Dataset
DATASET_EN = load_dataset('wikiann', 'en')
DATASET_DA = load_dataset('wikiann', 'da')

# Metrics
METRIC = evaluate.load("seqeval")
NER_FEATURE = DATASET_DA["train"].features["ner_tags"]
LABEL_NAMES = NER_FEATURE.feature.names
LABELS = DATASET_DA["train"][0]["ner_tags"]
LABELS = [LABEL_NAMES[i] for i in LABELS]


def main():
    raw_dataset_train = concatenate_datasets([DATASET_EN["train"], DATASET_DA["train"]])
    raw_dataset_val = concatenate_datasets([DATASET_EN["validation"], DATASET_DA["validation"]])
    tokenized_dataset_train = raw_dataset_train.map(
        tokenize_and_align_labels,
        batched=True,
        remove_columns=DATASET_DA["train"].column_names,
    )
    tokenized_dataset_val = raw_dataset_val.map(
        tokenize_and_align_labels,
        batched=True,
        remove_columns=DATASET_DA["validation"].column_names,
    )
        
    data_collator = DataCollatorForTokenClassification(tokenizer=TOKENIZER)
    
    id2label = {i: label for i, label in enumerate(LABEL_NAMES)}
    label2id = {v: k for k, v in id2label.items()}

    model = AutoModelForTokenClassification.from_pretrained(
        CHECKPOINT,
        id2label=id2label,
        label2id=label2id,
    )
    #print(model.config.num_labels)
    
    args = TrainingArguments(
        output_dir=os.path.join(get_root_path(), 'data', 'models', 'xlm-roberta-ner', 'checkpoint'),
        evaluation_strategy="epoch",
        save_strategy="epoch",
        learning_rate=2e-5,
        num_train_epochs=3,
        weight_decay=0.01,
        push_to_hub=False,
    )
    
    trainer = Trainer(
        model=model,
        args=args,
        train_dataset=tokenized_dataset_train,
        eval_dataset=tokenized_dataset_val,
        data_collator=data_collator,
        compute_metrics=compute_metrics,
        tokenizer=TOKENIZER,
    )
    trainer.train()
    trainer.save_model(output_dir=os.path.join(get_root_path(), 'data', 'models', f'{CHECKPOINT}-ner'))


def compute_metrics(eval_preds):
    logits, labels = eval_preds
    predictions = np.argmax(logits, axis=-1)

    # Remove ignored index (special tokens) and convert to labels
    true_labels = [[LABEL_NAMES[l] for l in label if l != -100] for label in labels]
    true_predictions = [
        [LABEL_NAMES[p] for (p, l) in zip(prediction, label) if l != -100]
        for prediction, label in zip(predictions, labels)
    ]
    all_metrics = METRIC.compute(predictions=true_predictions, references=true_labels)
    return {
        "precision": all_metrics["overall_precision"],
        "recall": all_metrics["overall_recall"],
        "f1": all_metrics["overall_f1"],
        "accuracy": all_metrics["overall_accuracy"],
    }    


def align_labels_with_tokens(labels, word_ids):
    new_labels = []
    current_word = None
    for word_id in word_ids:
        if word_id != current_word:
            # Start of a new word!
            current_word = word_id
            label = -100 if word_id is None else labels[word_id]
            new_labels.append(label)
        elif word_id is None:
            # Special token
            new_labels.append(-100)
        else:
            # Same word as previous token
            label = labels[word_id]
            new_labels.append(label)

    return new_labels

def tokenize_and_align_labels(examples):
    tokenized_inputs = TOKENIZER(
        examples["tokens"], truncation=True, is_split_into_words=True
    )
    all_labels = examples["ner_tags"]
    new_labels = []
    for i, labels in enumerate(all_labels):
        word_ids = tokenized_inputs.word_ids(i)
        new_labels.append(align_labels_with_tokens(labels, word_ids))

    tokenized_inputs["labels"] = new_labels
    return tokenized_inputs


if __name__ == "__main__":
    main()
    #print(DATASET_DA.column_names)
    # print(DATASET_DA["train"].features)
    # print(DATASET_DA["train"][0])
    # inputs = TOKENIZER(DATASET_DA["train"][0]["tokens"], is_split_into_words=True)
    # print(inputs.tokens())
    # print(align_labels_with_tokens(DATASET_DA["train"][0]["ner_tags"], inputs.word_ids()))
    