import numpy as np
import timeit
import pandas as pd
import datetime
import argparse
import sys

from functools import partial
from nltk.tokenize.regexp import regexp_tokenize
from sklearn.model_selection import StratifiedShuffleSplit, cross_validate, GridSearchCV, StratifiedKFold
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics import confusion_matrix
from .keras_models import create_one_layer, create_dualInputSimple, create_dualInputLarge

from keras.wrappers.scikit_learn import KerasClassifier
from keras.layers import merge, Dense, Dropout, Input, concatenate
from keras.models import Model
from keras.models import Sequential
from keras.constraints import maxnorm
from keras.optimizers import Nadam

from keras.models import load_model

#tail -n 1000 Data/benign_badging_full_v2.txt > Data/benign_badging_1000.txt

def main():

    args = parse_arguments()

    perm_inputs, feat_inputs, comb_inputs, labels = vectorize(args["good_path"], args["mal_path"])
    perm_width = len(perm_inputs[0])
    feat_width = len(feat_inputs[0])
    comb_width = len(comb_inputs[0])


    final_test(args, perm_inputs, feat_inputs, comb_inputs, labels)

    return

def vectorize(good_path, mal_path):
    '''
    accepts the filepath to the benign and malicious data files. Then tokenizes
    according to regular expressions below and creates matching labels.
    '''

    with open(good_path) as f:
        ben_samples = f.readlines()
    with open(mal_path) as f:
        mal_samples = f.readlines()

    print("Benign Size: ")
    print(sys.getsizeof(ben_samples))
    print("Malware Size: ")
    print(sys.getsizeof(mal_samples))

    samples = ben_samples + mal_samples

    labels = np.array([])
    for x in ben_samples:
        labels = np.append(labels, 0)
    for x in mal_samples:
        labels = np.append(labels, 1)

    print("Samples: ")
    print(sys.getsizeof(samples))

    #regular expressions for each desired data type
    perm_pattern = "(?:\w|\.)+(?:permission).(?:\w|\.)+"
    feat_pattern = "(?:\w|\.)+(?:hardware).(?:\w|\.)+"
    comb_pattern = "(?:\w|\.)+(?:hardware|permission).(?:\w|\.)+"

    perm_vect = CountVectorizer(analyzer=partial(regexp_tokenize, pattern=perm_pattern))
    feat_vect = CountVectorizer(analyzer=partial(regexp_tokenize, pattern=feat_pattern))
    comb_vect = CountVectorizer(analyzer=partial(regexp_tokenize, pattern=comb_pattern))

    time0 = timeit.default_timer()
    #each vectorizer tokenizes via fit_transform() and then is converted to a dense vector
    perm_inputs_sparse = perm_vect.fit_transform(samples)
    perm_inputs_dense = perm_inputs_sparse.todense()
    perm_inputs = np.array(perm_inputs_dense)

    feat_inputs_sparse = feat_vect.fit_transform(samples)
    feat_inputs_dense = feat_inputs_sparse.todense()
    feat_inputs = np.array(feat_inputs_dense)

    comb_inputs_sparse = comb_vect.fit_transform(samples)
    comb_inputs_dense = comb_inputs_sparse.todense()
    comb_inputs = np.array(comb_inputs_dense)


    return perm_inputs, feat_inputs, comb_inputs, labels


def final_test(args, perm_inputs, feat_inputs, comb_inputs, labels):
    '''
    performs final test and validation across each train ratio on hardcoded
    hyperparemeter values established by the gridsearch
    '''
    perm_width = int(len(perm_inputs[0]))
    feat_width = int(len(feat_inputs[0]))
    comb_width = int(len(comb_inputs[0]))
    print ('perm width: ' + str(perm_width))
    input_ratios = args["input_ratio"]
    
    mod = args["model"]
    print(mod)
    size = 32

    model = load_model('/home/pi/DeepLearning/Models/' + args['load'])
    print(model)
    #models = {'oneLayer_comb':oneLayer_comb, 'oneLayer_perm':oneLayer_perm, \
    #'oneLayer_feat':oneLayer_feat, 'dual_simple':dual_simple, 'dual_large':dual_large}
    #models = ('oneLayer_comb', 'oneLayer_feat', 'oneLayer_perm', 'dual_simple', 'dual_large')


    for m in mod:
        print (m)
        data = []
        for r in args["train_ratio"]:
            percent=float(r)/100
            #stratified shuffle split used for cross validation
            sss = StratifiedShuffleSplit(n_splits=5, random_state=0, test_size=1-percent)
            cm = np.zeros([2,2], dtype=np.int64)
            train_time = 0.0
            test_time = 0.0
            ir = 0
            for train_index, test_index in sss.split(perm_inputs, labels):
                perm_train, perm_test = perm_inputs[train_index], perm_inputs[test_index]
                feat_train, feat_test = feat_inputs[train_index], feat_inputs[test_index]
                comb_train, comb_test = comb_inputs[train_index], comb_inputs[test_index]
                labels_train, labels_test = labels[train_index], labels[test_index]

                if m == "oneLayer_comb":
                    print ('oneLayer_comb')
                    epoch = 32
                    batch = 32
                    time1 = timeit.default_timer()
                    labels_pred = model.predict(comb_test, batch_size=batch)
                    time2 = timeit.default_timer()

                elif m == "oneLayer_perm":
                    print ('oneLayer_perm')
                    batch = 32
                    epoch = 16
                    time1 = timeit.default_timer()
                    print (time1-time0)
                    labels_pred = model.predict(perm_test, batch_size=batch)
                    time2 = timeit.default_timer()
                    print (time2-time1)

                elif m == "oneLayer_feat":
                    print ('oneLayer_feat')
                    batch = 16
                    epoch = 32
                    time1 = timeit.default_timer()
                    labels_pred = model.predict(feat_test, batch_size = batch)
                    time2 = timeit.default_timer()

                elif m == "dual_simple":
                    print ('dual_simple')
                    batch = 16
                    epoch = 32
                    ir = .125
                    print("args: batch=%i, epochs=%i, ir=%f, perm_width=%i, feat_width=%i" % (batch, epoch, ir, perm_width, feat_width))
                    print (type(perm_width))
                    print (type(feat_width))
                    time1 = timeit.default_timer()
                    labels_pred = model.predict([perm_test, feat_test], batch_size=batch)
                    time2 = timeit.default_timer()

                elif m == "dual_large":
                    print ('dual_large')
                    batch = 128
                    epoch = 32
                    ir = .125
                    time1 = timeit.default_timer()
                    labels_pred = model.predict([perm_test, feat_test], batch_size=batch)
                    time2 = timeit.default_timer()


                #train_time += time1-time0
                test_time += time2-time1
                labels_pred = (labels_pred > 0.5)
                cm = cm + confusion_matrix(labels_test, labels_pred)
            acc = calc_accuracy(cm)
            prec = calc_precision(cm)
            rec = calc_recall(cm)
            f1 = calc_f1(prec, rec)
            avg_train_time = train_time/5
            avg_test_time = test_time/5

        data.append(dict(zip(["model_name", "neurons", "train_ratio", "input_ratio", \
        "epochs", "batch_size", "accuracy", "precision", "recall", "f1_score", \
        "avg_train_time", "avg_test_time"], \
        [m, size, r, ir, epoch, batch, acc, prec, rec, f1, avg_train_time, avg_test_time])))


        print ('saving results for model: ' + str(m))
        save_results(data, m)



def save_results(data, modelName):
    d = datetime.datetime.today()
    month = str( '%02d' % d.month)
    day = str('%02d' % d.day)
    hour = str('%02d' % d.hour)
    year = str('%02d' % d.year)
    min = str('%02d' % d.minute)

    df = pd.DataFrame(data)
    try:
        path1 = '/home/pi/DeepLearning/Results/deepResults/multi_input/fall18/loaded/' + modelName + month + day + year + '-' + hour + min + '.csv'
        file1 = open(path1, "w+")
    except:
        path1 = "gridSearch" + modelName + ".csv"
        file1 = open(path1, "w+")
    df.to_csv(file1, index=False)
    file1.close()

    return 0

def calc_accuracy(cm):
    TP = float(cm[1][1])
    TN = float(cm[0][0])
    n_samples = cm.sum()
    return (TP+TN)/n_samples

def calc_precision(cm):
    TP = float(cm[1][1])
    FP = float(cm[0][1])
    return TP/(TP+FP)

def calc_recall(cm):
    TP = float(cm[1][1])
    FN = float(cm[1][0])
    return TP/(FN+TP)

def calc_f1(precision, recall):
    return 2*((precision*recall)/(precision+recall))


def parse_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument("-gp", "--good_path", help="Good File Path")
    parser.add_argument("-mp", "--mal_path", help="Malware File Path")
    
    parser.add_argument("--load", help="Learning Model File Name")

    parser.add_argument("-e", "--epochs", help="Number of Epochs, can be list for grid search", type=int, nargs="*")
    parser.add_argument("-tr", "--train_ratio", nargs="*", type=int,
                        help="Set Test Ratios. Enter as a percent (20,40,60,80). Can be a list space delimited")
    parser.add_argument("-bs", "--batch_size", nargs="*", type=int,
                        help="Batch size. Can be a list space delimited")
    parser.add_argument("-n", "--neurons", nargs="*", type=int,
                        help="Number of Neurons. Can be a list space delimited")
    parser.add_argument("-o", "--optimizer", nargs="*",
                        help="Optimizers. Can be a list space delimited")
    parser.add_argument("-w", "--weight_constraint", nargs="*", type=int,
                        help="Weight Constraint. Can be a list space delimited")
    parser.add_argument("-d", "--dropout", nargs="*", type=int,
                        help="Dropout. Enter as percent (10,20,30,40...). Can be a list space delimited.")
    parser.add_argument("-model", "--model", help="Select which model to run: oneLayer_perm, \
     oneLayer_feat, oneLayer_comb, dual_large, or dual_simple")
    parser.add_argument("-s", "--splits", help="Number of Splits for SSS", type=int)
    parser.add_argument("-ir", "--input_ratio", help="ratio of layer width between \
     features and permissions layers", type=float, nargs="*")

    args = parser.parse_args()

    arguments = {}

    if args.good_path:
        good_path = args.good_path
        arguments["good_path"] = good_path
    else:
        print("Needs Good Path with -gp or --good_path")
        sys.exit()

    if args.mal_path:
        mal_path = args.mal_path
        arguments["mal_path"] = mal_path
    else:
        print("Needs Malware Path with -mp or --mal_path")
        sys.exit()

    if args.load:
        load = args.load
        arguments["load"] = load
    else:
        print("Needs Model File with --load")
        sys.exit()

    if args.model in ["oneLayer_perm", "oneLayer_feat", "oneLayer_comb", \
     "dual_large", "dual_simple"]:
        model = [args.model]
    else:
        print("Needs Model of type: oneLayer_perm, oneLayer_feat, oneLayer_comb, dual_large, or dual_simple")
        sys.exit()
    arguments["model"] = model

    if args.epochs:
        epochs = args.epochs
    else:
        print("Defaulting to 16 epochs")
        epochs = [16]
    arguments["epochs"] = epochs
    if args.train_ratio:
        train_ratio = args.train_ratio
    else:
        print("Defaulting to testing all ratios")
        train_ratio = [20, 40, 60, 80]
    arguments["train_ratio"] = train_ratio

    if args.batch_size:
        batch_size = args.batch_size
    else:
        print("Defaulting to Batch Size 10")
        batch_size = [10]
    arguments["batch_size"] = batch_size

    if args.neurons:
        neurons = args.neurons
    else:
        print("Defaulting to 32 Neurons")
        neurons = [32]
    arguments["neurons"] = neurons

    if args.optimizer:
        optimizer = args.optimizer
    else:
        print("Defaulting to NADAM Optimizer")
        optimizer = "Nadam"
    arguments["optimizer"] = optimizer

    if args.weight_constraint:
        weight_constraint = args.weight_constraint
    else:
        print("Defaulting to weight constraint 5")
        weight_constraint = [5]
    arguments["weight_constraint"] = weight_constraint

    if args.dropout:
        dropout = args.dropout
    else:
        print("Defaulting to dropout of 10%")
        dropout = [10]
    arguments["dropout"] = dropout

    if args.splits:
        splits = args.splits
    else:
        print("Defaulting to 5 SSS Split")
        splits = [5]
    arguments["splits"] = splits

    if args.input_ratio:
        input_ratio = args.input_ratio
    else:
        print("default to .25 input ratio")
        input_ratio = [.25]
    arguments["input_ratio"] = input_ratio


    return arguments


if __name__ == "__main__":
    main()
