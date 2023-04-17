# Can take csv and json files as input

import os
import csv
import re
import json
import nltk

from nlp.utils.utility import get_root_path

nltk.download('stopwords')
nltk.download('punkt')

DATA_PATH = os.path.join(get_root_path(), 'data')
TRAIN_PATH = os.path.join(DATA_PATH, 'word2vec')
EXTRA_STOPWORDS_PATH = os.path.join(get_root_path(), 'msc', 'resources', 'stopwords.txt')


def read_json(json_path):
    with open(json_path, 'r') as f:
        json_file = json.load(f)
        sentences = nltk.sent_tokenize(json_file['text'])
        tokens = [preprocess_text(sentence) for sentence in sentences]
        return tokens
    
def preprocess_text(text: str):
    stopwords = nltk.corpus.stopwords.words('danish')
    with open(EXTRA_STOPWORDS_PATH, 'r') as f:
        stopwords.extend(f.read().splitlines())
    stopwords = set(stopwords)
    # remove numbers and special characters
    text = re.sub("[^A-Za-z]+", " ", text)
    tokens = nltk.word_tokenize(text)
    tokens = [token.lower() for token in tokens if token.lower() not in stopwords]
    return tokens
    
def read_article_body(filename):
    tokens = []
    with open(filename, newline='') as csvfile:
        reader = csv.DictReader(csvfile)
        for row in reader:
            try:
                sentences = nltk.sent_tokenize(row['article_body'])
                tokens.extend([preprocess_text(sentence) for sentence in sentences])
            except (UnicodeDecodeError, UnicodeEncodeError):
                pass
    return tokens

def load_data(path = TRAIN_PATH):
    files = os.listdir(path)
    tokens = []
    for file in files:
        if file.endswith('.json'):
            #pass
            tokens.extend(read_json(os.path.join(path, file)))
        elif file.endswith('.csv'):
            tokens.extend(read_article_body(os.path.join(path, file)))
    return tokens

    
if __name__ == '__main__':
    tokens = load_data()
    print(tokens[0:50])