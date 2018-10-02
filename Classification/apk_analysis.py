import numpy as np
import timeit
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
    good_path = '/home/josh/Documents/COSC/research/ml_malware/malware_data/goodPermissionsFinal.txt'
    mal_path = '/home/josh/Documents/COSC/research/ml_malware/malware_data/malwarePermissionsFinal.txt'

    with open(good_path) as f:
        gdprm = f.readlines()
    with open(mal_path) as f:
        mlprm = f.readlines()

    perms = gdprm + mlprm

    labels = np.array([])
    for x in gdprm:
        labels = np.append(labels, 0)
    for x in mlprm:
        labels = np.append(labels, 1)


    count_vect = CountVectorizer(input=u'content', analyzer=u'word', token_pattern='name=\'[^\']*\'')
    time0 = timeit.default_timer()
    features = count_vect.fit_transform(perms)
    time1 = timeit.default_timer() #time to tokenize
    print type(features)
    print features.get_shape()
    #print count_vect.get_feature_names()
    print 'tokenize time: ' + str(time1-time0)
    print '\n'

    #proportion of data to test on vs total
    ratios = [.8, .6, .4, .2]

    print "BernoulliNB"
    for x in ratios:
        BNclf = BernoulliNB()
        test_model(BNclf, features, labels, x)
        print'\n'
    print '---------------------------\n'
    print "MultiNomialNB"
    for x in ratios:
        NBclf = MultinomialNB()
        test_model(NBclf, features, labels, x)
        print '\n'
    print '---------------------------\n'
    print "DecisionTree"
    for x in ratios:
        DTclf = DecisionTreeClassifier(min_samples_split = 20)
        test_model(DTclf, features, labels, x)
        print '\n'
    print '---------------------------\n'
    print "LogisticRegression"
    for x in ratios:
        LRclf = LogisticRegression(C=10, solver='lbfgs')
        test_model(LRclf, features, labels, x)
        print '\n'


    #print "SVM"
    #SVclf = SVC(C=10.0, kernel='rbf')
    #SVclf.fit(features_train, labels_train)
    #pred3 = SVclf.predict(features_test)
    #acc3 = accuracy_score(labels_test, pred3)

    #print acc3
    ''' alternative to shuffle_split
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

def test_model(model, features, labels, test_size):
    '''
    recieves an instance of an untrained model plus features and labels. performs
    training on 5 splits of data at ratio specified by test size.
    '''
    print 'training ratio: ' + str(1-test_size)
    sss = StratifiedShuffleSplit(n_splits=5, test_size=test_size)
    avg_acc = 0.0
    avg_true_pos = 0.0
    avg_true_neg = 0.0
    avg_fpos = 0.0
    avg_fneg = 0.0
    avg_train_time = 0.0
    avg_test_time = 0.0
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
        f_pos = conf_mat[0][1]
        f_neg = conf_mat[1][0]
        avg_acc += float(correct)/test_len
        avg_true_pos += conf_mat[0][0]
        avg_true_neg += conf_mat[1][1]
        avg_fpos += float(f_pos)/test_len
        avg_fneg += float(f_neg)/test_len
        avg_train_time += time1-time0
        avg_test_time += time2-time1
        #print conf_mat
        ''' more verbose output
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
    avg_acc = avg_acc/5
    avg_true_pos = avg_true_pos/5
    avg_true_neg = avg_true_neg/5
    avg_fpos = avg_fpos/5
    avg_fneg = avg_fneg/5
    avg_train_time = avg_train_time/5
    avg_test_time = avg_test_time/5
    print 'accuracy measures: '
    print 'avg acc: ' + str(avg_acc)
    print 'avg_true_pos: ' + str(avg_true_pos)
    print 'avg_true_neg: ' + str(avg_true_neg)
    print 'avg fpos: ' + str(avg_fpos)
    print 'avg fneg: ' + str(avg_fneg)

    print '\nruntime measures: '
    print 'avg train time: ' + str(avg_train_time)
    print 'avg test time: ' + str(avg_test_time)
    print '\n'
    return

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


if __name__ == "__main__":
    main()
