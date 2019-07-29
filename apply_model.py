import sys
import getopt

from os import listdir
from os.path import join
from pathlib import Path

import tensorflow as tf

from evaluation_utils.cross_validation import run_cross_validation
from evaluation_utils.scores import base_scores, get_main_scores, macro_average

def main(argv):
    params = {
        'dim_char': 50,
        'char_lstm_size': 25,
        'embedding_dim': 200,
        'lstm_size': 100,
        'dropout': 0.5,
        'batch_size': 20,
        'epochs': 30,
    }
    data_location = 'data/goldstandard_e_wikipedia-pubmed-and-PMC-w2v_ed_200'
    data_name = 'data'
    validation_mode = '10_cval_random_d_True_0.5'
    train = True
    evalu = True
    predi = True

    try:
        opts, args = getopt.getopt(argv, "hd:s:v:", ["data=", "set=", "val=", "train=", "eval=", "predi=", "dim_char=", "char_lstm_size=", "embedding_dim=", "lstm_size=", "dropout=", "batch_size=", "epochs="])
    except getopt.GetoptError:
        print('Usage: python apply_model.py --data=<location> --set=<name> --val=<dir> opt: --train --eval --predi --dim_char --char_lstm_size --embedding_dim --lstm_size --dropout --batch_size --epochs')
        sys.exit(2)
    for opt, arg in opts:
        if opt == '-h':
            print('Usage: python apply_model.py --data=<location> --set=<name> --val=<dir> opt: --train --eval --predi --dim_char --char_lstm_size --embedding_dim --lstm_size --dropout --batch_size --epochs')
            sys.exit()
        elif opt in ("-d", "--data"):
            data_location = arg
        elif opt in ("-s", "--set"):
            data_name = arg
        elif opt in ("-v", "--val"):
            validation_mode = arg
        elif opt in ("--train"):
            train = bool(int(arg))
        elif opt in ("--eval"):
            evalu = bool(int(arg))
        elif opt in ("--predi"):
            predi = bool(int(arg))
        else:
            o = opt.strip('-')
            if o in params.keys():
                params[o] = type(params[o])(arg)

    with Path(join(data_location, data_name+'.tag.txt')).open() as lmap, Path(join(data_location, data_name+'.chars.txt')).open() as cmap:
        label_mapping = {line.strip(): idx for idx, line in enumerate(lmap)}
        char_mapping = {line.strip(): idx for idx, line in enumerate(cmap)}

    params['nchars'] = len(char_mapping)
    params['num_tags'] = len(label_mapping)

    config = {
        'embedding': join(data_location, data_name + '_embedding.npz'),
        'tags': join(data_location, data_name + '.tag.txt'),
        'data_location': data_location,
        'data_name': data_name,
        'validation_mode': validation_mode
    }

    model, result = run_cross_validation(params, config, label_mapping, do_train=train, do_eval=evalu, do_predict=predi)

    cross_val_scores = base_scores(result, save_location=data_location)
    df = get_main_scores(cross_val_scores)
    strict_mean, strict_std = macro_average(df.loc['strict',:], save_location=data_location)
    ent_type_mean, ent_type_std = macro_average(df.loc['ent_type',:], save_location=data_location)
    print("""
        Evaluation had the following results for the cross validation:
        Partial Recognition:
        Mean Precision: {} (std {}),
        Mean Recall: {} (std {}),
        Mean FScore: {} (std {}).
        Exact Recognition:
        Mean Precision: {} (std {}),
        Mean Recall: {} (std {}),
        Mean FScore: {} (std {}).""".format(round(ent_type_mean.precision, 2), 
                                            round(ent_type_std.precision, 2),
                                            round(ent_type_mean.recall, 2), 
                                            round(ent_type_std.recall, 2),
                                            round(ent_type_mean.f1, 2),
                                            round(ent_type_std.f1, 2),
                                            round(strict_mean.precision, 2), 
                                            round(strict_std.precision, 2), 
                                            round(strict_mean.recall, 2), 
                                            round(strict_std.recall, 2), 
                                            round(strict_mean.f1, 2), 
                                            round(strict_std.f1, 2)))

if __name__ == "__main__":
   main(sys.argv[1:]) 