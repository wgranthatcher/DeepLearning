from __future__ import division
import numpy as np
import timeit
import pandas
import datetime
from functools import partial
from nltk.tokenize.regexp import regexp_tokenize

from sklearn.naive_bayes import MultinomialNB
from sklearn.naive_bayes import BernoulliNB
from sklearn.linear_model import LogisticRegression
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.metrics import confusion_matrix
from sklearn.tree import DecisionTreeClassifier


def main():
    '''
    Notes for running:
    - written for python 2.7 -> change print statements if using 3
    - required deps -> install scikit learn (google it)
    - edit filepaths
    '''
    input_type = 'permissions'

    good_path = '/home/josh/Documents/COSC/Research/APK_project/apk_repo/test_sets/large/v2/mal_badging_full_v2.txt'
    mal_path = '/home/josh/Documents/COSC/Research/APK_project/apk_repo/test_sets/large/v2/benign_badging_full_v2.txt'
    results_dir = '/home/josh/Documents/COSC/Research/APK_project/DeepLearningResearch/Results/shallowResults/imbalanced-'


    with open(good_path) as f:
        gdprm = f.readlines()
    with open(mal_path) as f:
        mlprm = f.readlines()

    features = gdprm + mlprm

    labels = np.array([])
    for x in gdprm:
        labels = np.append(labels, 0)
    for x in mlprm:
        labels = np.append(labels, 1)

    token_pattern = None
    if input_type == 'hardware':
        #token_pattern = 'android\.hardware\.[^\']*'
        token_pattern = "(?:\w|\.)+(?:hardware).(?:\w|\.)+"
    elif input_type == 'permissions':
        #token_pattern = 'android\.permission\.[^\']*'
        #token_pattern = "(?<=name=\')[^(?:p)]*(?:permission)[^\']*"
        token_pattern = "(?:\w|\.)+(?:permission).(?:\w|\.)+"
    else:
        #token_pattern = 'android\.(?:hardware|permission)\.[^\']*'
        #token_pattern = "(?<=name=\')[^(?:p|h)]*(?:permission|hardware)[^\']*"
        token_pattern = "(?:\w|\.)+(?:permission|hardware).(?:\w|\.)+"
    print token_pattern

    #count_vect = CountVectorizer(input=u'content', analyzer=u'word', token_pattern=token_pattern)
    count_vect = CountVectorizer(analyzer=partial(regexp_tokenize, pattern=token_pattern))

    time0 = timeit.default_timer()
    data_features = count_vect.fit_transform(features)
    time1 = timeit.default_timer() #time to tokenize
    print type(features)
    print data_features.get_shape()
    #for x in count_vect.get_feature_names():
    #    print x
    print 'tokenize time: ' + str(time1-time0)
    print '\n'

    #proportion of data to test on vs total
    ratios = [.8, .6, .4, .2]
    columns = ['avg_acc', 'fpos_rate', 'fneg_rate', 'precision', 'recall',
    'f1_score', 'avg_test_time', 'avg_train_time']
    indices = [.2,.4,.6,.8]
    print "BernoulliNB"
    bNBdf=pandas.DataFrame(columns=columns)
    print bNBdf
    for x in ratios:
        model_name="BernoulliNB"
        BNclf = BernoulliNB()
        bNBdf = test_model(bNBdf, BNclf, data_features, labels, x)
        results_to_csv(bNBdf, model_name, results_dir, input_type)
        print'\n'
    print '---------------------------\n'
    print "MultiNomialNB"
    mnNBdf=pandas.DataFrame(columns=columns)#, index=indices)
    for x in ratios:
        model_name = "MultinomialNB"
        NBclf = MultinomialNB()
        mnNBdf = test_model(mnNBdf, NBclf, data_features, labels, x)
        results_to_csv(mnNBdf, model_name, results_dir, input_type)
        print '\n'
    print '---------------------------\n'
    print "DecisionTree"
    dtdf=pandas.DataFrame(columns=columns)#, index=indices)
    for x in ratios:
        model_name = "DecisionTree"
        DTclf = DecisionTreeClassifier()#min_samples_split = 20)
        dtdf = test_model(dtdf, DTclf, data_features, labels, x)
        results_to_csv(dtdf, model_name, results_dir, input_type)
        print '\n'
    print '---------------------------\n'
    print "LogisticRegression"
    lgdf=pandas.DataFrame(columns=columns)#, index=indices)
    for x in ratios:
        model_name = "Logistic_Regression"
        LRclf = LogisticRegression(C=10, solver='lbfgs')
        lgdf = test_model(lgdf, LRclf, data_features, labels, x)
        results_to_csv(lgdf, model_name, results_dir, input_type)
        print '\n'

    '''# alternative to shuffle_split
    print "stratifiedKFolds"
    skf = StratifiedKFold(n_splits=5, shuffle=True)
    for train_index, test_index in skf.split(features, labels):
        X_train, X_test = features[train_index], features[test_index]
        y_train, y_test = labels[train_index], labels[test_index]

        BNclf = BernoulliNB()
        BNclf.fit(X_train, y_train)
        pred = BNclf.predict(X_test)
        print accuracy_score(y_test, pred)
    '''

def test_model(data_frame, model, features, labels, test_size):
    '''
    recieves an instance of an untrained model plus features and labels. performs
    training on 5 splits of data at ratio specified by test size.
    '''
    print 'training ratio: ' + str(1-test_size)
    sss = StratifiedShuffleSplit(n_splits=5, test_size=test_size)
    avg_acc = 0.0
    true_pos = 0.0
    true_neg = 0.0
    false_pos = 0.0
    false_neg = 0.0
    avg_fpos_rate = 0.0
    avg_tpos_rate = 0.0
    avg_tneg_rate = 0.0
    avg_fneg_rate = 0.0
    avg_train_time = 0.0
    avg_test_time = 0.0
    precision = 0.0
    recall = 0.0
    f1_score = 0.0
    for train_index, test_index in sss.split(features, labels):
        X_train, X_test = features[train_index], features[test_index]
        y_train, y_test = labels[train_index], labels[test_index]

        time0 = timeit.default_timer()
        model.fit(X_train, y_train)
        time1 = timeit.default_timer()
        pred = model.predict(X_test)
        time2 = timeit.default_timer()
        conf_mat = confusion_matrix(y_test, pred)
        test_len = len(y_test)
        correct = conf_mat[0][0] + conf_mat[1][1]
        false_neg = conf_mat[0][1]
        false_pos = conf_mat[1][0]
        true_neg = conf_mat[0][0]
        true_pos = conf_mat[1][1]

        #debug
        #print str(false_pos) + ' ' + str(true_pos)

        precision += float(true_pos)/float(true_pos+false_pos)
        print 'precision ' + str(precision)
        recall += float(true_pos)/float(true_pos+false_neg)
        print 'recall ' + str(recall)
        #f1_score += 2*((precision*recall)/(precision+recall))
        #print 'f1_score ' + str(f1_score)

        avg_acc += float(correct)/test_len
        avg_fpos_rate += float(false_pos)/test_len
        avg_fneg_rate += float(false_neg)/test_len
        avg_tpos_rate += float(true_pos)/test_len
        avg_tneg_rate += float(true_neg)/test_len
        avg_train_time += time1-time0
        avg_test_time += time2-time1
        '''
        more verbose output
        print 'samples tested: ' + str(test_len)
        print 'correct predictions: ' + str(correct)
        print 'false positives: ' + str(f_pos)
        print 'false negatives: ' + str(f_neg)
        print '--------'
        print 'acc rate: ' + str(float(correct)/test_len)
        print 'false positive rate: ' + str(float(f_pos)/test_len)
        print 'false negative rate: ' + str(float(f_neg)/test_len)
        print '\n\n'
        '''
    avg_acc = avg_acc/5.0
    avg_tpos_rate = avg_tpos_rate/5.0
    avg_tneg_rate = avg_tneg_rate/5.0
    avg_fpos_rate = avg_fpos_rate/5.0
    avg_fneg_rate = avg_fneg_rate/5.0
    avg_train_time = avg_train_time/5.0
    avg_test_time = avg_test_time/5.0
    avg_precision = precision/5.0
    avg_recall = recall/5.0
    avg_f1_score = 2*((avg_precision*avg_recall)/(avg_precision+avg_precision))

    #append to dataframe
    loc = 1-test_size
    data_frame.loc[str(loc)] = pandas.Series({'avg_acc':avg_acc, 'fpos_rate':avg_fpos_rate,
    'fneg_rate':avg_fneg_rate, 'precision':avg_precision, 'recall':avg_recall,
    'f1_score':avg_f1_score, 'avg_test_time':avg_test_time, 'avg_train_time':avg_train_time})
    print(data_frame)


    print 'accuracy measures: '
    print 'avg acc: ' + str(avg_acc)
    print 'avg fpos rate: ' + str(avg_fpos_rate)
    print 'avg fneg rate: ' + str(avg_fneg_rate)

    print '\nruntime measures: '
    print 'avg train time: ' + str(avg_train_time)
    print 'avg test time: ' + str(avg_test_time)
    print '\n'
    return data_frame

#below function not used
def full_results(labels, predictions):
    i = 0
    correct = 0
    false_pos = 0
    false_neg = 0
    results = acc_stats()
    while(i < len(labels)):
        if(labels[i] == predictions[i]):
            results.inc_correct()
        elif(labels[i] == 1 and predictions[i] == 0):
            results.inc_fneg()
        elif(labels[i] == 0 and predictions[i] == 1):
            results.inc_fpos()
        else:
            print 'shouldnt happen'
        i+=1
    results.print_stats()

def results_to_csv(data_frame, model_name, target_dir, input_type):
    '''
    accepts pandas dataframe and writes it to a csv in dest directory
    '''
    d = datetime.datetime.today()
    month = str('%02d' % d.month)
    day = str('%02d' % d.day)
    hour = str('%02d' % d.hour)
    min = str('%02d' % d.minute)

    try:
        out_target = target_dir + model_name + '_' + input_type + '_' + month + '-' + day + '-' + hour + min + '.csv'
        print out_target
        outfile = open(out_target,'w+')
    except:
        print 'HIT EXCEPTION'
        out_target = model_name + month + day + hour + min + '.csv'
        outfile = open(out_target, 'w+')
    data_frame.to_csv(outfile, index=True)
    outfile.close()
    return




if __name__ == "__main__":
    main()
