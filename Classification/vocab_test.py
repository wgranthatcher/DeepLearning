import numpy as np
import timeit
import pandas
import datetime

from functools import partial
from nltk.tokenize.regexp import regexp_tokenize
from sklearn.model_selection import StratifiedShuffleSplit, cross_validate, GridSearchCV, StratifiedKFold
from sklearn.feature_extraction.text import CountVectorizer

def main():

    data_dir = '/home/jmcgiff/Documents/research/DeepLearningResearch/Data/'
    #ben_data = data_dir + 'badging_med/ben_badging_med.txt'
    #mal_data = data_dir + 'badging_med/mal_badging_med.txt'
    mal_data = data_dir + 'mal_badging_full_v2.txt'
    ben_data = data_dir + 'benign_badging_full_v2.txt'
    old_ben = data_dir + 'goodPermissionsFinal.txt'
    old_bal = data_dir + 'malwarePermissionsFinal.txt'

    vectorize(ben_data, old_ben, ben_data, old_ben)

def vectorize(good_path, old_ben, mal_path, old_mal):

    with open(good_path) as f:
        ben_new = f.readlines()
    with open(mal_path) as f:
        mal_new = f.readlines()
    with open(old_ben) as f:
        ben_old = f.readlines()
    with open(old_mal) as f:
        mal_old = f.readlines

    new_samples = ben_new + mal_new
    old_samples = ben_old + ben_new

    perm_pattern = "(?:\w|\.)+(?:permission).(?:\w|\.)+"
    feat_pattern = "(?:\w|\.)+(?:hardware).(?:\w|\.)+"
    comb_pattern = "(?:\w|\.)+(?:hardware|permission).(?:\w|\.)+"
    old_pattern =  "(\\b(:?uses-|optional-)?permission:\s[^\s]*)"

    new_perm_vect = CountVectorizer(analyzer=partial(regexp_tokenize, pattern=perm_pattern))
    old_vect = CountVectorizer(analyzer=partial(regexp_tokenize, pattern=old_pattern))
    comb_vect = CountVectorizer(analyzer=partial(regexp_tokenize, pattern=comb_pattern))
    feat_vect = CountVectorizer(analyzer=partial(regexp_tokenize, pattern=feat_pattern))


    time0 = timeit.default_timer()
    new_perms = new_perm_vect.fit_transform(new_samples)
    feats = feat_vect.fit_transform(new_samples)
    comb = comb_vect.fit_transform(new_samples)



    #old_inputs = old_vect.fit_transform(old_samples)

    print "new perms vocab:" + str(new_perms.shape)
    print "comb vocab: " + str(comb.shape)
    print "feat vocab: " + str(feats.shape)
    #print "old perms vocab:" + str(old_inputs.shape)



    #old_vocab_out = open('/home/jmcgiff/Documents/research/DeepLearningResearch/Data/vocab_files/old_vocab.txt', "w+")
    #new_vocab_out = open('/home/jmcgiff/Documents/research/DeepLearningResearch/Data/vocab_files/new_vocab.txt', "w+")

    #for x in new_vect.get_feature_names():
#        new_vocab_out.write("%s\n" % str(x))
#    for x in old_vect.get_feature_names():
#        old_vocab_out.write("%s\n" % str(x))
    return

if __name__ == "__main__":
    main()
