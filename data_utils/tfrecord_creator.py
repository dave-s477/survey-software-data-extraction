# -*- coding: utf-8 -*-
import random 
import tensorflow.train as train

from pathlib import Path
from tensorflow.python_io import TFRecordWriter
from tensorflow.data import TFRecordDataset
from os.path import join, exists
from os import makedirs, listdir, remove

def count_samples(file_list, output_location):
    """
    Count the number of sentences in BIO files.
    
    Arguments:
        file_list {[str]} -- [list of BIO fies]
        output_location {[str]} -- [path to BIO files]
    
    Returns:
        [int],[int] -- [num of sentences], [num of words]
    """
    sample_counter = 0
    word_counter = 0
    for f in file_list:
        with open(join(output_location, f), 'r') as datafile:
            for line in datafile:
                if line in ['\n','\r\n']:
                    sample_counter += 1
                elif not line.startswith("-DOCSTART-") and '---' not in line:
                    word_counter += 1
    return sample_counter, word_counter

def make_or_clean_if_exists(*paths):
    for path in paths:
        if not exists(path):
            makedirs(path)
        else: 
            for f in listdir(path):
                remove(join(path, f))
 
def random_n_fold_crossval_tfrec_format(n_folds, output_location, file_list=['BIO_format.txt'], base_name='data', downsample=True, downsampling_rate=0.5):
    """
    Create Tensorflow Records from BIO format. 
    For a n-fold cross validation n tfrecs are created. 
    The creation is based on counting the number of samples 
    in the file and randomly sorting them to n buckets.

    A training and test version are created, respectively 
    for each fold.
    Downsampling is applied as a very simple batch control
    on the training version. 
    
    Arguments:
        n_folds {[int]} -- [num of folds for cross validation]
        output_location {[str]} -- [path to data]
    
    Keyword Arguments:
        file_list {list} -- [BIO files] (default: {['BIO_format.txt']})
        base_name {str} -- [base name for reading and writing] (default: {'data'})
        downsample {bool} -- [apply downsampling to corpus] (default: {False})
        downsampling_rate {float} -- [rate for downsampling empty data] (default: {0.0})
    """
    random.seed(1)
    sample_count, word_count = count_samples(file_list, output_location)
    print("Creating TFRecord for {}-fold random cross validation.".format(n_folds))
    print("Found {} input samples (sentences) and {} words.".format(sample_count, word_count))

    train_path = join(output_location, str(n_folds) + '_cval_random_d_' + str(downsample) + "_" + str(downsampling_rate), 'records_train')
    train_path_labels = join(output_location, str(n_folds) + '_cval_random_d_' + str(downsample) + "_" + str(downsampling_rate), 'labels_train')
    test_path = join(output_location, str(n_folds) + '_cval_random_d_' + str(downsample) + "_" + str(downsampling_rate), 'records_test')
    test_path_labels = join(output_location, str(n_folds) + '_cval_random_d_' + str(downsample) + "_" + str(downsampling_rate), 'labels_test')
    make_or_clean_if_exists(train_path, train_path_labels, test_path, test_path_labels)

    with Path(join(output_location, (base_name + '.words.txt'))).open() as f:
        word_to_idx = {line.strip(): idx for idx, line in enumerate(f)}
    with Path(join(output_location, (base_name + '.tag.txt'))).open() as f:
        label_mapping = {line.strip(): idx for idx, line in enumerate(f)}
    with Path(join(output_location, (base_name + '.chars.txt'))).open() as f:
        char_mapping = {line.strip(): idx for idx, line in enumerate(f)}

    tfwriter_set_train = []
    correct_labels_train = []
    sentences_to_labels_train = []
    words_to_labels_train = []
    tfwriter_set_test = []
    correct_labels_test = []
    sentences_to_labels_test = []
    words_to_labels_test = []
    for idx in range(n_folds):
        record_name = base_name + "_" + str(idx+1) + ".tfrec"
        tfwriter_set_train.append(TFRecordWriter(join(train_path, record_name)))
        correct_labels_train.append([])
        sentences_to_labels_train.append([])
        words_to_labels_train.append([])
        tfwriter_set_test.append(TFRecordWriter(join(test_path, record_name)))
        correct_labels_test.append([])
        sentences_to_labels_test.append([])
        words_to_labels_test.append([])

    plain_words = []
    for f in file_list:
        print("Working on file: " + join(output_location, f))
        with open(join(output_location, f), 'r') as datafile:
            indices = []
            labels = []
            words = []
            high_low = []
            length = []
            for line in datafile:
                if line in ['\n','\r\n']:
                    index_list = train.FeatureList(
                        feature=[train.Feature(int64_list=train.Int64List(value=[ind])) for ind in indices])
                    label_list = train.FeatureList(
                        feature=[train.Feature(int64_list=train.Int64List(value=[lab])) for lab in labels])
                    word_list = train.FeatureList(
                        feature=[train.Feature(bytes_list=train.BytesList(value=[wor.encode()])) for wor in words])
                    upper_list = train.FeatureList(
                        feature=[train.Feature(int64_list=train.Int64List(value=[hl])) for hl in high_low])
                    len_list = train.FeatureList(
                        feature=[train.Feature(int64_list=train.Int64List(value=[le])) for le in length])
                    sentences = train.FeatureLists(feature_list={
                        'indices': index_list,
                        'labels': label_list,
                        'plain': word_list,
                        'upper': upper_list,
                        'length': len_list
                    })
                    example = train.SequenceExample(feature_lists=sentences)
                    sentence_reconstructed = '_'.join(plain_words)
                    labels_in_sent = list(set(labels))
                    set_to_put = random.randint(0,n_folds-1)
                    tfwriter_set_test[set_to_put].write(example.SerializeToString())
                    correct_labels_test[set_to_put].extend(labels)
                    for i in range(len(words)):
                        sentences_to_labels_test[set_to_put].append(sentence_reconstructed)
                    words_to_labels_test[set_to_put].extend(plain_words)
                    if downsample and len(labels_in_sent) == 1 and labels_in_sent[0] == 5 and random.random() < downsampling_rate:
                        pass
                    else:
                        tfwriter_set_train[set_to_put].write(example.SerializeToString())
                        correct_labels_train[set_to_put].extend(labels)
                        for i in range(len(words)):
                            sentences_to_labels_train[set_to_put].append(sentence_reconstructed)
                        words_to_labels_train[set_to_put].extend(plain_words)
                    indices = []
                    labels = []
                    words = []
                    plain_words = []
                    high_low = []
                    length = []

                elif not line.startswith("-DOCSTART-") and '---' not in line:
                    info = line.split()
                    word = info[0]
                    characters = [char_mapping[x] for x in word]
                    words.append(" ".join(str(x) for x in characters))
                    labels.append(label_mapping[info[1]])
                    
                    if info[0] not in word_to_idx:  
                        indices.append(len(word_to_idx))
                        print(info[0])
                        raise RuntimeError("Tried to process a word that is not in the prebuilt dictionary.")
                    else:
                        indices.append(word_to_idx[info[0]])
                        
                    high_low.append(word.isupper())
                    length.append(len(word))
                    plain_words.append(word)

    for idx in range(n_folds):
        label_name = base_name + "_" + str(idx+1) + ".tfrec" + '_labels'
        with open(join(train_path_labels, label_name), 'w') as cor_labels:
            print("In train " + str(idx+1) + " are " + str(len(correct_labels_train[idx])) + " samples (words).")
            if len(correct_labels_train[idx]) != len(sentences_to_labels_train[idx]) or len(correct_labels_train[idx]) != len(words_to_labels_train[idx]):
                raise(RuntimeError("Sentence length does not match labels."))
            for l,s,w in zip(correct_labels_train[idx], sentences_to_labels_train[idx], words_to_labels_train[idx]):
                cor_labels.write(str(l) + " " + s + " " + w + '\n')
        tfwriter_set_train[idx].close()

        with open(join(test_path_labels, label_name), 'w') as cor_labels:
            print("In test " + str(idx+1) + " are " + str(len(correct_labels_test[idx])) + " samples (words).")
            if len(correct_labels_test[idx]) != len(sentences_to_labels_test[idx]) or len(correct_labels_test[idx]) != len(words_to_labels_test[idx]):
                raise(RuntimeError("Sentence length does not match labels.\n"))
            for l,s,w in zip(correct_labels_test[idx], sentences_to_labels_test[idx], words_to_labels_test[idx]):
                cor_labels.write(str(l) + " " + s + " " + w + '\n')
        tfwriter_set_test[idx].close()
    print("Done building the TFRecords.")