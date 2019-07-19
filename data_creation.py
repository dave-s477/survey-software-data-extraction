import tensorflow as tf
import numpy as np
import json
import sys
import re
import getopt

from os.path import join, exists
from os import makedirs, listdir, remove

from data_utils.build_embedding import build_embedding
from data_utils.build_vocab import build_vocab
from data_utils.BIO_tokenizer import Tokenizer
from data_utils.tfrecord_creator import random_n_fold_crossval_tfrec_format
from data_utils.preprocess import preprocess_token, preprocess_complete, preprocess_simple

def build_dataset(n_folds, word_emb, emb_size, data_path, data_files, rebuild_level, downsample, downsampling_rate): 
    """
    Build a dataset for testing a Tensorflow model on extraction of 
    Software and databases in the bioNerDS corpus.
    Creates BIO-format, vocabulary, relevant embedding vectors and
    TFRecords as seperate outputs. 

    TFRecords are used for Training.
    
    Arguments:
        word_emb {[str]} -- [name of word embedding to use]
        emb_size {[int]} -- [embedding dimension of used word embedding]
        data_path {[str]} -- [base path to data directory]
        data_files {[str]} -- [data files, at least one]
        rebuild_level {[str]} -- [data can be build from intermediate steps]
    """
    if '/' in word_emb:
        emb_loc = join('word_embeddings', word_emb.split('/')[-1])
    else:
        emb_loc = join('word_embeddings', word_emb)
    if not exists(emb_loc):
        raise(RuntimeError("Word Embedding to use does not exist in 'word_embeddings'."))
    output_directory = join('data', data_path + "_e_" + word_emb.split(".")[0] + "_ed_" + str(emb_size))
    if not exists(output_directory):
        makedirs(output_directory)
    build_data_flag = build_vocab_flag = build_embedding_flag = build_tfrecord_flag = False
    if rebuild_level == 'data': 
        build_data_flag = build_vocab_flag = build_embedding_flag = build_tfrecord_flag = True
    elif rebuild_level == 'vocab':
        build_vocab_flag = build_embedding_flag = build_tfrecord_flag = True
    elif rebuild_level == 'embedding':
        build_embedding_flag = build_tfrecord_flag = True
    elif rebuild_level == 'record':
        build_tfrecord_flag = True
    else: 
        pass

    if build_data_flag: 
        tok = Tokenizer(data_path, data_files)
        tok.load_data()
        tok.unescape_text()
        tok.preprocess_text()
        tok.split_to_sentences()
        tok.process_data()
        tok.write_BIO(output_directory)
    if build_vocab_flag:
        build_vocab(output_directory)
    if build_embedding_flag:
        build_embedding(emb_loc, emb_size, output_directory)
    if build_tfrecord_flag:
        random_n_fold_crossval_tfrec_format(n_folds, output_directory, downsample=downsample, downsampling_rate=downsampling_rate)
    
    return output_directory

def main(argv):
    n_folds = 10
    rebuild = 'data'
    downsample = True
    rate = 0.5
    word_emb = "wikipedia-pubmed-and-PMC-w2v.bin"
    emb_size = 200
    cleanup = True
    try:
        opts, args = getopt.getopt(argv, "hn:b:d:r:w:e:c:", ["build=", "down=", "rate=", "wordemb=", "embdim=", "clean="])
    except getopt.GetoptError:
        print('Usage: python data_creation.py -n=10 --clean=1 -b/--build=<rebuild level: data/vocab/embedding/record> -d/--down=<use downsampling: 1/0 - True/False> -r/--rate=<downsampling rate: 0.0 - 1.0> -w/--wordemb=<name (not path)> -e/--embdim=200')
        sys.exit(2)
    for opt, arg in opts:
        if opt == '-h':
            print('Usage: python data_creation.py -n=10 --clean=1 -b/--build=<rebuild level: data/vocab/embedding/record> -d/--down=<use downsampling: 1/0 - True/False> -r/--rate=<downsampling rate: 0.0 - 1.0> -w/--wordemb=<name (not path)> -e/--embdim=200')
            sys.exit()
        elif opt in ("-n"):
            n_folds = arg
        elif opt in ("-b", "--build"):
            rebuild = arg
        elif opt in ("-d", "--down"):
            downsample = bool(arg)
        elif opt in ("-r", "--rate"):
            rate = float(arg)
        elif opt in ("-w", "--wordemb"):
            word_emb = arg
        elif opt in ("-e", "--embdim"):
            emb_size = int(arg)
        elif opt in ("-c", "--clean"):
            cleanup = bool(arg)
    
    data_path = 'goldstandard'
    data_files = ['devel.human.ascii.html', 'gb.human.ascii.html', 'next.5.human.ascii.html', 'eval.human.ascii.html', 'eval.set2.25.human.html' ]
    loc = build_dataset(n_folds, word_emb, emb_size, data_path, data_files, rebuild, downsample=downsample, downsampling_rate=rate)
    if cleanup:
        for f in listdir(loc):
            if f.startswith('data') or f.startswith('BIO_'):
                remove(join(loc, f))

if __name__ == "__main__":
   main(sys.argv[1:])                