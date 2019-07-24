import numpy as np
import tensorflow as tf
import pickle
import json

from tf_metrics import precision, recall, f1
from pathlib import Path
from os import listdir

tf.logging.set_verbosity(tf.logging.INFO)

# The model design was copied from https://github.com/guillaumegenthial/sequence_tagging

def lstm_model(features, labels, mode, params):
    """
    Defines bi-LSTM-CRF model with character bi-LSTM.
    Also defines behaviour for training/testing and prediction.
    
    Arguments:
        features {[type]} -- [input features to the network]
        labels {[type]} -- [labels for supervised learning]
        mode {[type]} -- [training mode]
        params {[type]} -- [hyper-parameter configuration]
    """
    with Path(params['tags']).open() as f:
        # 'O' denotes the negative class
        indices = [idx for idx, tag in enumerate(f) if tag.strip() != 'O']
    
    training = (mode == tf.estimator.ModeKeys.TRAIN)

    K = tf.get_variable(name="char_embeddings", dtype=tf.float32, shape=[params['nchars'], params['dim_char']])
    char_embeddings = tf.nn.embedding_lookup(K, features['plain'])
    char_embeddings = tf.layers.dropout(char_embeddings, rate=params['dropout'], training=training)

    # Reshaping the character input to be LSTM conform. From:
    # [ batch_size (sentences) , words (padded for the batch) , characters (also padded) , char_embedding_size ]
    # to:
    # [ batch_size (sentences) x words (padded for the batch) , characters (also padded) , char_embedding_size ]
    s = tf.shape(char_embeddings) 
    char_embeddings = tf.reshape(char_embeddings, shape=[-1, s[-2], s[-1]])
    char_embeddings.set_shape([None, None, params['dim_char']])

    # Actual sequence length is also separately passed to the LSTM 
    word_length = tf.reshape(features['length'], shape=[-1])
    # Calculating word-wise character representations by using a bi-lstm
    char_cell_fw = tf.contrib.rnn.LSTMCell(params['char_lstm_size'], state_is_tuple=True)
    char_cell_bw = tf.contrib.rnn.LSTMCell(params['char_lstm_size'], state_is_tuple=True)
    _, ((_, char_output_fw), (_, char_output_bw)) = tf.nn.bidirectional_dynamic_rnn(char_cell_fw,
                                                                            char_cell_bw, 
                                                                            char_embeddings, 
                                                                            sequence_length=word_length,
                                                                            dtype=tf.float32, scope='chars')
    # HERE: concatenation of final states
    char_output = tf.concat([char_output_fw, char_output_bw], axis=-1)
    char_rep = tf.reshape(char_output, shape=[-1, s[1], 2*params['char_lstm_size']])
    
    # Get the pre-selected word vectors as numpy array
    embedding = np.load(params['embedding'])['embeddings'] 
    L = np.vstack([embedding, [[0.] * params['embedding_dim']]])
    # Word embeddings are not trainable
    L = tf.Variable(L, dtype=tf.float32, trainable=False)
    pretrained_embeddings = tf.nn.embedding_lookup(L, features['indices'])
    word_embeddings = tf.concat([pretrained_embeddings, char_rep], axis=-1)
    word_embeddings = tf.layers.dropout(word_embeddings, rate=params['dropout'], training=training)
    
    # Sentence-wise bi-LSTM
    cell_fw = tf.contrib.rnn.LSTMCell(params['lstm_size'])
    cell_bw = tf.contrib.rnn.LSTMCell(params['lstm_size'])
    sequence_lengths = tf.reshape(features['sequence_length'], shape=[-1])
    (output_fw, output_bw), _ = tf.nn.bidirectional_dynamic_rnn(cell_fw,
                                                                cell_bw, 
                                                                word_embeddings, 
                                                                sequence_length=sequence_lengths,
                                                                dtype=tf.float32, scope='words')
    # HERE: concatenation of output for each individual word
    context_rep = tf.concat([output_fw, output_bw], axis=-1)
    context_rep = tf.layers.dropout(context_rep, rate=params['dropout'], training=training)
    
    # Mapping the output to number of classes and scoring by CRF
    logits = tf.layers.dense(context_rep, params['num_tags'])
    crf_params = tf.get_variable("crf", [params['num_tags'], params['num_tags']], dtype=tf.float32)
    pred_ids, _ = tf.contrib.crf.crf_decode(logits, crf_params, sequence_lengths)
        
    weights = tf.sequence_mask(sequence_lengths)
    if mode == tf.estimator.ModeKeys.PREDICT:
        predictions = {
            'pred_class': pred_ids,
            'mask': weights
        } 
        return tf.estimator.EstimatorSpec(mode, predictions=predictions)
    
    log_likelihood, _ = tf.contrib.crf.crf_log_likelihood(logits, labels, sequence_lengths, crf_params)
    loss = tf.reduce_mean(-log_likelihood)
    
    # Using an external metric by https://github.com/guillaumegenthial/tf_metrics
    metrics = {
        'acc': tf.metrics.accuracy(labels, pred_ids, weights),
        'precision': precision(labels, pred_ids, params['num_tags'], indices, weights),
        'recall': recall(labels, pred_ids, params['num_tags'], indices, weights),
        'f1': f1(labels, pred_ids, params['num_tags'], indices, weights),
    }
    for metric_name, op in metrics.items():
        tf.summary.scalar(metric_name, op[1])
    if mode == tf.estimator.ModeKeys.EVAL:
        return tf.estimator.EstimatorSpec(mode, loss=loss, eval_metric_ops=metrics)
    if mode == tf.estimator.ModeKeys.TRAIN:
        train_op = tf.train.AdamOptimizer().minimize(loss, global_step=tf.train.get_or_create_global_step())
        return tf.estimator.EstimatorSpec(mode, loss=loss, train_op=train_op)

def tfrec_data_input_fn(filenames, num_epochs=1, batch_size=64, shuffle=False, repeat=False):
    """
    Create a reader/feature generator for TFRecords.
    
    Arguments:
        filenames {[str]} -- [(multiple) TFRecord File name]
    
    Keyword Arguments:
        num_epochs {int} -- [training epochs] (default: {1})
        batch_size {int} -- [batch size] (default: {64})
        shuffle {bool} -- [shuffle samples] (default: {False})
        repeat {bool} -- [repeat samples] (default: {False})
    
    Returns:
        [fnct] -- [generator function]
    """
    def _input_fn():
        def _parse_record(tf_record):
            features = {
                'indices': tf.FixedLenSequenceFeature([], dtype=tf.int64, allow_missing=True),
                'labels': tf.FixedLenSequenceFeature([], dtype=tf.int64, allow_missing=True),
                'plain': tf.FixedLenSequenceFeature([], dtype=tf.string, allow_missing=True),
                'upper': tf.FixedLenSequenceFeature([], dtype=tf.int64, allow_missing=True),
                'length': tf.FixedLenSequenceFeature([], dtype=tf.int64, allow_missing=True)}
            _,record = tf.parse_single_sequence_example(tf_record, sequence_features=features)

            indices = tf.cast(record['indices'], tf.int32)
            labels = tf.cast(record['labels'], tf.int32)
            upper = tf.cast(record['upper'], tf.int32)
            length = tf.cast(record['length'], tf.int32)
            plain = tf.strings.split(record['plain'])
            plain = tf.sparse_tensor_to_dense(plain, default_value='0')
            plain = tf.strings.to_number(plain, out_type=tf.int32)
            sequence_length = [tf.size(labels)]
            
            return { 
                'indices': indices, 
                'plain': plain, 
                'length': length, 
                'upper': upper,
                'sequence_length': sequence_length}, labels
        
        dataset = tf.data.TFRecordDataset(filenames)
        dataset = dataset.map(_parse_record)

        if repeat:
            dataset = dataset.repeat(num_epochs)
        if shuffle:
            dataset = dataset.shuffle(buffer_size=256, seed=1)

        dataset = dataset.padded_batch(batch_size, padded_shapes=({'indices': [None], 
                                                                     'plain': [None, None],
                                                                     'length': [None],
                                                                     'upper': [None],
                                                                     'sequence_length': [None]}, [None]))

        iterator = dataset.make_one_shot_iterator()
        features, labels = iterator.get_next()

        return features, labels
    
    return _input_fn