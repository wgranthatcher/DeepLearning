import numpy as np
import timeit
import time
import pydot, graphviz

from functools import partial
from nltk.tokenize.regexp import regexp_tokenize
from sklearn.model_selection import StratifiedShuffleSplit, cross_validate, GridSearchCV, StratifiedKFold
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics import confusion_matrix

from keras.wrappers.scikit_learn import KerasClassifier
from keras.layers import Merge, Dense, Dropout, Input, concatenate
from keras.models import Model
from keras.models import Sequential
from keras.constraints import maxnorm
from keras.optimizers import Nadam
from keras.utils import plot_model


def main():
    good_path = "/home/jmcgiff/Documents/research/DeepLearningResearch/Data/badging_med/mal_badging_med.txt"

    mal_path = "/home/jmcgiff/Documents/research/DeepLearningResearch/Data/badging_med/ben_badging_med.txt"

    tr = .80
    neurons = 32
    batch = 32
    epochs = 8

    perm_inputs, feat_inputs, labels = vectorize(good_path, mal_path)
    perm_width = int(len(perm_inputs[0]))
    feat_width = int(len(feat_inputs[0]))
    cm = np.zeros([2,2], dtype=np.int64)
    model = create_dualInputLarge(input_ratio=.125, neurons=neurons, perm_width=perm_width, \
    feat_width=feat_width)
    plot_model(model, to_file='model.png')
    model.summary()
    time.sleep(10)

    sss = StratifiedShuffleSplit(n_splits=5, random_state=0, test_size=1-tr)
    i = 0
    for train_index, test_index in sss.split(perm_inputs, labels):
        perm_train, perm_test = perm_inputs[train_index], perm_inputs[test_index]
        feat_train, feat_test = feat_inputs[train_index], feat_inputs[test_index]
        labels_train, labels_test = labels[train_index], labels[test_index]

        model = create_dualInputLarge(input_ratio=.125, neurons=neurons, perm_width=perm_width, \
        feat_width=feat_width)

        print('\nsplit %i' %i)
        model.fit([perm_train, feat_train], labels_train, epochs=epochs, batch_size=batch)
        labels_pred = model.predict([perm_test, feat_test], batch_size=batch)
        labels_pred = (labels_pred > 0.5)
        cm = cm + confusion_matrix(labels_test, labels_pred)
        i += 1

    acc = calc_accuracy(cm)
    print 'average accuracy was: ' + str(acc)

    return

def calc_accuracy(cm):
    TP = float(cm[1][1])
    TN = float(cm[0][0])
    n_samples = cm.sum()
    return (TP+TN)/n_samples

def vectorize(good_path, mal_path):

    with open(good_path) as f:
        ben_samples = f.readlines()
    with open(mal_path) as f:
        mal_samples = f.readlines()

    samples = ben_samples + mal_samples

    labels = np.array([])
    for x in ben_samples:
        labels = np.append(labels, 0)
    for x in mal_samples:
        labels = np.append(labels, 1)

    perm_pattern = "(?:\w|\.)+(?:permission).(?:\w|\.)+"
    feat_pattern = "(?:\w|\.)+(?:hardware).(?:\w|\.)+"

    perm_vect = CountVectorizer(analyzer=partial(regexp_tokenize, pattern=perm_pattern))
    feat_vect = CountVectorizer(analyzer=partial(regexp_tokenize, pattern=feat_pattern))

    perm_inputs_sparse = perm_vect.fit_transform(samples)
    perm_inputs_dense = perm_inputs_sparse.todense()
    perm_inputs = np.array(perm_inputs_dense)

    feat_inputs_sparse = feat_vect.fit_transform(samples)
    feat_inputs_dense = feat_inputs_sparse.todense()
    feat_inputs = np.array(feat_inputs_dense)

    return perm_inputs, feat_inputs, labels

def create_dualInputLarge(input_ratio, feat_width, perm_width, neurons=32, dropout_rate=0.1):
    '''this model performs additional analysis with layers after concatenation'''
    perm_width=int(perm_width)
    perm_input = Input(shape=(perm_width,), name='permissions_input')
    x = Dense(neurons, activation='relu')(perm_input)
    x = Dropout(dropout_rate)(x)
    x = Dense(neurons, activation='relu')(x)
    feat_input = Input(shape=(feat_width,), name='features_input')
    y = Dense(int(neurons*input_ratio), activation='relu')(feat_input)
    x = concatenate([x, y])
    x = Dense(int((neurons+(neurons*input_ratio))/2), activation='relu')(x)
    x = Dropout(dropout_rate)(x)
    x = Dense(int((neurons+(neurons*input_ratio))/2), activation='relu')(x)
    output = Dense(1, activation='sigmoid', name="output")(x)
    model = Model(inputs=[perm_input, feat_input], outputs=output)
    model.compile(loss='binary_crossentropy', optimizer='nadam', metrics=['accuracy'])
    return model

if __name__ == "__main__":
    main()
