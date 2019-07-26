# -*- coding: utf-8 -*-
import numpy as np
import gensim

from pathlib import Path
from os.path import join

def build_embedding(embedding_model, embedding_size, output_dir, base_name='data'):
    """
    Create Embedding Vectors for a given dictionary and pre-trained word embedding. 
    The Embedding Vectors have of course already been created in pre-training.
    But to provide Tensorflow with the vectors it is easiert to extract the ones actually needed (memory issue).
    
    Arguments:
        embedding_model {[str]} -- [location of the embedding]
        embedding_size {[int]} -- [vector size of the embedding]
        output_dir {[str]} -- [data location]
    
    Keyword Arguments:
        base_name {str} -- [naming for input vocab and output emb] (default: {'data'})
    """
    word_file = base_name + '.words.txt'
    with Path(join(output_dir, word_file)).open() as f:
        word_to_idx = {line.strip(): idx for idx, line in enumerate(f)}
    size_vocab = len(word_to_idx)

    embeddings = np.zeros((size_vocab + 1, embedding_size))

    found = 0
    if embedding_model.endswith('.bin'):
        word2vec = gensim.models.KeyedVectors.load_word2vec_format(embedding_model, binary=True)
    elif embedding_model.endswith('.txt'):
        word2vec = gensim.models.KeyedVectors.load_word2vec_format(embedding_model)
    else:
        raise(RuntimeError("Word embedding has to be in text or binary format and also named that way."))

    with Path(join(output_dir, word_file)).open() as f:
        for line in f:
            word = line.strip()
            id_word = word_to_idx[word]
            if word in word2vec.vocab:
                word_vector = word2vec.wv[word]
                embeddings[id_word] = word_vector
                found += 1
            else:
                pass
            
    print('- done. Found {} vectors for {} words'.format(found, size_vocab))

    embedding_file = base_name + '_embedding.npz'
    np.savez_compressed(join(output_dir, embedding_file), embeddings=embeddings)