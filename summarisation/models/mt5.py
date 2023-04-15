import os
import json

from transformers import AutoTokenizer
from datasets import Dataset

from summarisation.utils.utility import get_root_path

CHECKPOINT = "google/mt5-small"
PREFIX = "summarize: "
tokenizer = AutoTokenizer.from_pretrained(CHECKPOINT)


def create_train_test_split():
    train = []
    test = []
    data_path = os.path.join(get_root_path(), 'data', 'json')
    train_len = round(len(os.listdir(data_path)) * 0.7)
    
    for idx, pdf in enumerate(os.listdir(data_path)):
        with open(os.path.join(data_path, pdf), 'r') as f:
            data = json.load(f)
            if idx <= train_len:
                train.append(data)
            else:
                test.append(data)
    return {"train": train, "test": test}


def preprocess_function(examples):

    inputs = [PREFIX + doc for doc in examples["text"]]

    model_inputs = tokenizer(inputs, max_length=1024, truncation=True)

    labels = tokenizer(text_target=examples["summary"], max_length=128, truncation=True)

    model_inputs["labels"] = labels["input_ids"]

    return model_inputs
    
if __name__ == "__main__":
    dataset = create_train_test_split()
    print(dataset[0])
