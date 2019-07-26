import tensorflow as tf
import numpy as np
import json
import csv
import numbers
import warnings

from pathlib import Path
from os.path import join, exists
from os import makedirs, listdir, remove

from models.bi_lstm_crf import lstm_model, tfrec_data_input_fn

tf.logging.set_verbosity(tf.logging.INFO)

def model_description_string(params):
    """
    Create a string identifying the used model.
    
    Arguments:
        params {[dict]} -- [parameter configuration of the model]
    
    Returns:
        [str] -- [a single string describing the parameter setup]
    """
    model_description_string = 'model'
    for key in params:
        if key != 'epochs' and isinstance(params[key], numbers.Number):
            model_description_string = model_description_string + '_' + key + str(params[key])
    return model_description_string

def cross_validation(model, model_name, params, config, do_train=True, do_eval=True, do_predict=True):
    """
    Run a n-fold cross validation.
    
    Arguments:
        model {[type]} -- [model function to use]
        model_name {[type]} -- [description string for model]
        params {[type]} -- [hyper-parameter configuraiton for model]
        config {[type]} -- [base configuration parameters]
    
    Keyword Arguments:
        do_train {bool} -- [whether the model is trained] (default: {True})
        do_eval {bool} -- [whether the tensorflow evaluation is done] (default: {True})
        do_predict {bool} -- [whether prediction testing is applied] (default: {True})
    """
    results_dir = join(config['data_location'], 'results', model_name, 'epoch_' + str(params['epochs']))
    trained_model_dir = join(config['data_location'], 'trained_models', model_name)
    if not exists(results_dir):
        makedirs(results_dir)
    if not exists(trained_model_dir):
        makedirs(trained_model_dir)

    prediction_matrix = {}
    params.update(config)
    validation_dir = join(config['data_location'], config['validation_mode'])
    train_data = join(validation_dir, 'records_train')
    data_segmentation = sorted(listdir(train_data), key=lambda x: int(x.split('_')[-1].split('.')[0]))
    print("Starting the validation on a {}-fold validation".format(len(data_segmentation)))
    for test_set in data_segmentation:
        if len(data_segmentation) <= 1:
            print("Only one set, training and testing on same data.")
            training_set = [test_set]
        else: 
            training_set = [join(train_data, d) for d in data_segmentation if d != test_set]

        classifier = tf.estimator.Estimator(model_fn=model, model_dir=join(trained_model_dir, test_set), params=params)

        test_location = join(validation_dir, 'records_test')
        if do_train:
            print("Starting training with evaluation set {}".format(test_set))
            tfrec_train_input_fn = tfrec_data_input_fn(training_set, batch_size=params['batch_size'], num_epochs=params['epochs'], shuffle=True, repeat=True)
            classifier.train(input_fn=tfrec_train_input_fn)

        if do_eval:
            print("Starting evaluation on {}".format(test_set))
            tfrec_test_input_fn = tfrec_data_input_fn([join(test_location, test_set)], params['batch_size'])
            eval_result = classifier.evaluate(input_fn=tfrec_test_input_fn)
            for key in eval_result:
                eval_result[key] = float(eval_result[key])
            res_file_name = test_set.split('.tfrec')[0] + '.json'
            with open(join(results_dir, res_file_name), 'w') as result_file:
                json.dump(eval_result, result_file, sort_keys=True, indent=4)

        if do_predict:
            print("Starting prediction on {}".format(test_set))
            prediction_matrix[test_set] = [[],[]]
            tfrec_pred_input_fn = tfrec_data_input_fn([join(test_location, test_set)], 1)
            prediction_result = classifier.predict(input_fn=tfrec_pred_input_fn)
            for prediction in prediction_result:
                for pair in zip(prediction['pred_class'], prediction['mask']):
                    if pair[1]:
                        prediction_matrix[test_set][0].append(pair[0]) 

    return prediction_matrix, results_dir

def get_gold_labels(config, prediction_matrix):
    """
    Get the correct labels for the prediction.
    """
    validation_dir = join(config['data_location'], config['validation_mode'], 'labels_test')
    data_segmentation = sorted(listdir(validation_dir), key=lambda x: int(x.split('.')[0].split('_')[-1]))
    for test_set in data_segmentation:
        test_set_name = test_set.split('_labels')[0]
        prediction_matrix[test_set_name].append([])
        prediction_matrix[test_set_name].append([])
        with open(join(validation_dir, test_set), 'r') as label_file:
            for line in label_file:
                if not line in ['\n', '\r\n']:
                    line_split = line.split()
                    label = line_split[0]
                    prediction_matrix[test_set_name][1].append(label)
                    prediction_matrix[test_set_name][2].append(line_split[1])
                    prediction_matrix[test_set_name][3].append(line_split[2])
    return prediction_matrix

def build_output_format(config, model_name, prediction_matrix, label_mapping_reverse):
    """
    Write output of prediction to file and return sequence of truth and predictions. 
    """
    output_location = join(config['data_location'], 'output', model_name)
    if not exists(output_location):
        makedirs(output_location)
    predictions = {}
    correct_labels = {}
    for test_set in prediction_matrix:
        predictions[test_set] = []
        correct_labels[test_set] = []
        with open(join(output_location, test_set + '_results.csv'), 'w') as csv_file:
            wr = csv.writer(csv_file, quoting=csv.QUOTE_ALL)
            for idx, cor_label in enumerate(prediction_matrix[test_set][1]):
                predicted = prediction_matrix[test_set][0][idx]
                predictions[test_set].append(label_mapping_reverse[str(predicted)])
                correct_labels[test_set].append(label_mapping_reverse[str(cor_label)])
                wr.writerow([prediction_matrix[test_set][3][idx], label_mapping_reverse[str(cor_label)], label_mapping_reverse[str(predicted)]])
    return correct_labels, predictions
    
def run_cross_validation(params, config, label_mapping, do_train=True, do_eval=True, do_predict=True):
    """
    Wrapper around simple cross validation, which creates output of predictions.
    
    Arguments:
        params {[type]} -- [model configuration]
        config {[type]} -- [general configuration]
    
    Keyword Arguments:
        do_train {bool} -- [if the model is trained] (default: {True})
        do_eval {bool} -- [if model is evaluated by tensorflow] (default: {True})
        do_predict {bool} -- [if model is evaluated through prediction] (default: {True})
    
    Returns:
        [model, [truth, prediction]] -- [the model and the generated sequences]
    """
    model_name = model_description_string(params)
    model = lstm_model
    label_mapping_reverse = {}
    for key in label_mapping:
        label_mapping_reverse[str(label_mapping[key])] = key

    prediction_matrix, results_dir = cross_validation(model,
                                                      model_name,
                                                      params,
                                                      config,
                                                      do_train=do_train,
                                                      do_eval=do_eval,
                                                      do_predict=do_predict)
    truth = []
    pred = []
    if do_predict:  
        prediction_matrix = get_gold_labels(config, prediction_matrix)
        truth, pred = build_output_format(config, model_name, prediction_matrix, label_mapping_reverse)
    return model, [truth, pred]