import os

from gensim.models import Word2Vec

from nlp.data.word2vec_dataset import load_data
from nlp.utils.utility import get_root_path

def main():
    tokens = load_data()
    save_path = os.path.join(get_root_path(), 'data', 'models', 'word2vec')
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    model = Word2Vec(sentences=tokens, 
                     vector_size=100, 
                     max_vocab_size=50_000,
                     window=5, 
                     min_count=3,
                     sg=1, 
                     workers=-1)
    model.save(os.path.join(save_path, 'word2vec.model'))
    
if __name__ == '__main__':
    main()
