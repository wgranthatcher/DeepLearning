import numpy as np
import pandas
import sys
import argparse
import datetime

from .keras_models import create_binaryDecrease, create_fourDecrLayer, create_fourSameLayer, create_one_layer, create_thousand_layer
from keras.wrappers.scikit_learn import KerasClassifier
from sklearn.model_selection import StratifiedShuffleSplit, cross_validate, GridSearchCV
from sklearn.feature_extraction.text import CountVectorizer


def main():

    args = parse_arguments()

    # Vectorize the input
    features, labels = vectorize(args["good_path"], args["mal_path"], args["adverse"])

    # Grid Search
    if args["mode"] == "grid" :
        # for all models
        for m in args["model"] :
            # For all ratios
            for r in args["train_ratio"] :
                grid_search(m, features, labels, r, args)

    # Regular run -- Not Grid Search
    else :
        # For all models
        for m in args["model"] :
            # For all ratios
            for r in args["train_ratio"] :
                full_run(m, features, labels, r, args)

    return 0


# Method to vectorize the input data
def vectorize(good_path, mal_path, adverse):
    good_path = good_path
    mal_path = mal_path

    # read files
    with open(good_path) as f:
        gdprm = f.readlines()
    with open(mal_path) as f:
        mlprm = f.readlines()

    # Concatenate good and mal samples
    perms = gdprm + mlprm

    # append the labels
    # good is labeled 0
    # malware is labeled 1
    labels = np.array([])
    for x in gdprm:
        labels = np.append(labels, 0)
    for x in mlprm:
        labels = np.append(labels, 1)

    # Define the sklearn vectorizer
    count_vect = CountVectorizer(input=u'content', analyzer=u'word',
                                 token_pattern='(\\b(:?uses-|optional-)?permission:\s[^\s]*)')
    #time0 = timeit.default_timer()

    # vectorize input
    features = count_vect.fit_transform(perms)

    # convert to dense matrix
    features = features.todense()
    features = np.array(features)

    # This is in the case of adversarial learning
    # Some of the labels will be wrong on purpose
    if adverse:
        print("Adversarial Learning")
        # keep track of how many of each were changed
        count1 = 0
        count2 = 0

        gdprmsize = np.size(gdprm, 0)
        mlprmszie = np.size(mlprm, 0)

        # change 10% of the good labels
        for i in range(0, gdprmsize // 10):
            if labels[i] == 0:
                count1 += 1
                labels[i] = 1
        print("Good Permissions Changed: %d" % count1)

        # change 10% of the malware labels
        for i in range(gdprmsize, gdprmsize + mlprmszie // 10):
            if labels[i] == 1:
                count2 += 1
                labels[i] = 0
        print("Malware Permissions Changed: %d" % count2)

        total = count1 + count2
        print("Total Permissions Changed: %d" % total)

    print("Done Vectorizing Data")
    return features, labels


# Method for a standard test -- Not Grid Search
def full_run(modelName, features, labels, train_ratio, args):
    # Get Vars from input args
    epochs = args["epochs"][0]
    batch_size = args["batch_size"][0]
    neurons = args["neurons"][0]
    optimizer = args["optimizer"][0]
    weight_constraint = args["weight_constraint"][0]
    dropout_rate = args["dropout"][0]//100
    percent = float(train_ratio) / 100
    splits = args["splits"]

    #model_params = dict(batch_size=batch_size, epochs=epochs, neurons=neurons, optimizer=optimizer,
    #                    weight_constraint=weight_constraint, dropout_rate=dropout_rate)

    fit_params = dict(batch_size=batch_size, epochs=epochs)
    scoring = ['accuracy', 'precision', 'recall', 'f1']

    # Define and Build the Model based on input modelName
    if modelName == "oneLayer":
        model = KerasClassifier(build_fn=create_one_layer, batch_size=batch_size, epochs=epochs, neurons=neurons,
                                optimizer=optimizer, weight_constraint=weight_constraint, dropout_rate=dropout_rate,
                                verbose=2)
    elif modelName == "binaryDecrease":
        model = KerasClassifier(build_fn=create_binaryDecrease, batch_size=batch_size, epochs=epochs, neurons=neurons,
                                optimizer=optimizer, weight_constraint=weight_constraint, dropout_rate=dropout_rate,
                                verbose=2)
    elif modelName == "fourSame":
        model = KerasClassifier(build_fn=create_fourSameLayer, batch_size=batch_size, epochs=epochs, neurons=neurons,
                                optimizer=optimizer, weight_constraint=weight_constraint, dropout_rate=dropout_rate,
                                verbose=2)
    elif modelName == "fourDecr":
        model = KerasClassifier(build_fn=create_fourDecrLayer, batch_size=batch_size, epochs=epochs, neurons=neurons,
                                optimizer=optimizer, weight_constraint=weight_constraint, dropout_rate=dropout_rate,
                                verbose=2)
    #Added Thousand Layer  -WGH
    elif modelName == "thouLayer":
        model = KerasClassifier(build_fn=create_thousand_layer, batch_size=batch_size, epochs=epochs, neurons=neurons,
                                optimizer=optimizer, weight_constraint=weight_constraint, dropout_rate=dropout_rate,
                                verbose=2)

    # Shuffle split Definition for Cross Validation
    sss = StratifiedShuffleSplit(n_splits=splits, test_size=percent, random_state=0)

    # Running the model with Cross Validation
    cv_result = cross_validate(model, features, labels, cv=sss, fit_params=fit_params, return_train_score=True,
                               scoring=scoring, verbose=2)

    # Determine date for creating a file later
    # This helps to keep track of tests and prevents overwriting of results
    d = datetime.datetime.today()
    month = str( '%02d' % d.month)
    day = str('%02d' % d.day)
    hour = str('%02d' % d.hour)
    min = str('%02d' % d.minute)

    # saving the result of testing to a Pandas Dataframe
    df = pandas.DataFrame(cv_result)

    # Write the results out to a file
    try:
        path1 = '/home/lab309/pythonScripts/testResults/deep_results/' + modelName + month + day + hour + min + '.csv'
        file1 = open(path1, "a+")
    except:
        path1 = "results" + modelName + month + day + hour + min + ".csv"
        file1 = open(path1, "a+")
    df.to_csv(file1, index=True)
    file1.close()

    return 0


# Grid Search Method
def grid_search(modelName, features, labels, train_ratio, args):

    # Get Vars from input args
    splits = args["splits"]
    percent = float(train_ratio) / 100
    epochs = args["epochs"]
    batch_size = args["batch_size"]
    neurons = args["neurons"]
    optimizer = args["optimizer"]
    weight_constraint = args["weight_constraint"]
    dropout_rate = args["dropout"]

    # Define the grid based on params
    paramGrid = dict(epochs=epochs, batch_size=batch_size, optimizer=optimizer,
                     dropout_rate=dropout_rate, weight_constraint=weight_constraint,
                     neurons=neurons)

    # Model Definition based on input modelName
    if modelName == "oneLayer":
        model = KerasClassifier(build_fn=create_one_layer, verbose=0)
    elif modelName == "binaryDecrease":
        model = KerasClassifier(build_fn=create_binaryDecrease, verbose=0)
    elif modelName == "fourSame":
        model = KerasClassifier(build_fn=create_fourSameLayer, verbose=0)
    elif modelName == "fourDecr":
        model = KerasClassifier(build_fn=create_fourDecrLayer, verbose=0)

    # Define Split and Grid Search Cross Validation
    sss = StratifiedShuffleSplit(n_splits=splits, test_size=percent, random_state=0)
    grid = GridSearchCV(estimator=model, param_grid=paramGrid, n_jobs=1, cv=sss, refit=True, verbose=2)

    # Execute a grid search
    grid_fit = grid.fit(features, labels)

    # These are metrics that can be used later
    means = grid_fit.cv_results_['mean_test_score']
    stds = grid_fit.cv_results_['std_test_score']
    params = grid_fit.cv_results_['params']

    print("%s Best: %f using %s" % (modelName, grid_fit.best_score_, grid_fit.best_params_))

    # Determine date for creating a file later
    # This helps to keep track of tests and prevents overwriting of results
    d = datetime.datetime.today()
    month = str( '%02d' % d.month)
    day = str('%02d' % d.day)
    hour = str('%02d' % d.hour)
    min = str('%02d' % d.minute)

    # Save results to  Pandas Dataframe
    df = pandas.DataFrame(grid_fit.cv_results_)

    # Write the results out to a file
    try:
        path1 = '/home/lab309/pythonScripts/testResults/deep_results/gridSearch' + modelName + month + day + hour + min + '.csv'
        file1 = open(path1, "w+")
    except:
        path1 = "gridSearch" + modelName + ".csv"
        file1 = open(path1, "w+")
    df.to_csv(file1, index=True)
    file1.close()

    return 0


# Command Line Parameters are define in this method
def parse_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument("-gp", "--good_path", help="Good File Path")
    parser.add_argument("-mp", "--mal_path", help="Malware File Path")
    parser.add_argument("-ad", "--adverse", help="Turns on Adversarial Learning")
    parser.add_argument("-m", "--mode", help="Choose mode: full, grid")
    parser.add_argument("-e", "--epochs", help="Number of Epochs", type=int, nargs="*")
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
    parser.add_argument("-model", "--model", help="Select which model to run: all, one_layer, four_decr, four_same")
    parser.add_argument("-s", "--splits", help="Number of Splits for SSS", type=int)

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

    if args.adverse:
        adverse = True
    else:
        adverse = False
    arguments["adverse"] = adverse

    if args.mode == "grid":
        mode = "grid"
        print("Mode is %s" % mode)
    else:
        mode = "full"
        print("Mode is %s" % mode)
    arguments["mode"] = mode

    if args.model == "all":
        model = ["oneLayer", "fourDecr", "fourSame"]
    elif args.model in ["oneLayer", "fourDecr", "fourSame"]:
        model = [args.model]
    else:
        print("Defaulting to All models")
        model = ["oneLayer", "fourDecr", "fourSame"]
    arguments["model"] = model

    if args.epochs:
        epochs = args.epochs
    else:
        print("Defaulting to 16 epochs")
        epochs = 16
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
        batch_size = 10
    arguments["batch_size"] = batch_size

    if args.neurons:
        neurons = args.neurons
    else:
        print("Defaulting to 45 Neurons")
        neurons = 45
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
        weight_constraint = 5
    arguments["weight_constraint"] = weight_constraint

    if args.dropout:
        dropout = args.dropout
    else:
        print("Defaulting to dropout of 10%")
        dropout = 10
    arguments["dropout"] = dropout

    if args.splits:
        splits = args.splits
    else:
        print("Defaulting to 1 SSS Split")
        splits = 1
    arguments["splits"] = splits

    return arguments


if __name__ == "__main__":
    main()
