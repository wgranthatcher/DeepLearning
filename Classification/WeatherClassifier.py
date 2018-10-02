import numpy as np
import pandas
import argparse
import matplotlib.pyplot as plt
import datetime
from keras.models import Sequential
from keras.layers import Dense, Dropout, advanced_activations


def main():
    # Get input args
    args = parse_arguments()

    # Init random seed for reproducibility
    np.random.seed(0)

    # load the dataset
    dataframe = pandas.read_csv(args["data_path"], engine='python', parse_dates=['DATE'],
                                date_parser=lambda x: pandas.to_datetime(x, infer_datetime_format=True))

    # Define the training set using the input begin and end dates
    train_df= dataframe[(dataframe['DATE'] >= datetime.datetime(args["begin_train"],1,1)) &
                        (dataframe['DATE'] <= datetime.datetime(args["end_train"],12,31))]

    # Define the testing set using the input begin and end dates
    test_df = dataframe[(dataframe['DATE'] >= datetime.datetime(args["begin_test"],1,1)) &
                        (dataframe['DATE'] <= datetime.datetime(args["end_test"],12,31))]


    # Remove null and other invalid entries in the data
    train_data = np.nan_to_num(train_df['TAVG'].values.astype('float32'))
    test_data = np.nan_to_num(test_df['TAVG'].values.astype('float32'))

    # Combine the data to one array
    combined_data = np.append(train_data, test_data)

    # reshape dataset to window matrix
    look_back = 12  # This is the size of the window
    trainX, trainY = create_dataset(train_data, look_back)
    testX, testY = create_dataset(test_data, look_back)

    # Define and fit the model
    model = create_model(look_back=look_back)
    model.fit(trainX, trainY, epochs=500, batch_size=12, verbose=2)

    # Estimate model performance
    trainScore = model.evaluate(trainX, trainY, verbose=0)
    print('Train Score: %.2f MAE' % (trainScore))
    testScore = model.evaluate(testX, testY, verbose=0)
    print('Test Score: %.2f MAE' % (testScore))

    # generate predictions for training
    trainPredict = model.predict(trainX)
    testPredict = model.predict(testX)

    # shift train predictions for plotting
    trainPredictPlot = np.empty((len(combined_data), 1))
    trainPredictPlot[:] = np.nan
    trainPredictPlot[look_back:len(trainPredict) + look_back] = trainPredict

    # shift test predictions for plotting
    testPredictPlot = np.empty((len(combined_data), 1))
    testPredictPlot[:] = np.nan
    testPredictPlot[len(trainPredict) + (look_back * 2) + 1:len(combined_data) - 1] = testPredict

    # Combine the results
    combined_df = train_df.append(test_df)
    combined_dates = combined_df['DATE']

    # plot baseline and predictions
    plt.plot(combined_dates, combined_data, )
    plt.plot(combined_dates, trainPredictPlot)
    plt.plot(combined_dates, testPredictPlot)
    plt.minorticks_on()
    plt.show()


# Standard Model Creation
def create_model(look_back):
    # create and fit Multilayer Perceptron model
    model = Sequential()
    model.add(Dense(100, input_dim=look_back, activation='relu'))
    # model.add(Dense(50, activation='relu'))
    # model.add(Dense(25, activation='relu'))
    # model.add(Dense(10, activation='relu'))
    # model.add(Dense(5, activation='relu'))
    model.add(Dense(1))
    model.compile(loss='mean_absolute_error', optimizer='nadam')
    return model


# convert an array of values into a dataset matrix
def create_dataset(dataset, look_back=1):
    dataX, dataY = [], []
    for i in range(len(dataset) - look_back - 1):
        a = dataset[i:(i + look_back)]
        dataX.append(a)
        dataY.append(dataset[i + look_back])
    return np.array(dataX), np.array(dataY)


# Command Line Arguments are parsed here
def parse_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument("-dp", "--data_path", help="Data File Path")
    parser.add_argument("-ad", "--adverse", help="Turns on Adversarial Learning")
    parser.add_argument("-m", "--mode", help="Choose mode: full, grid")
    parser.add_argument("-e", "--epochs", help="Number of Epochs", type=int, nargs="*")
    parser.add_argument("-tr", "--train_ratio", nargs="*", type=int,
                        help="Set Train Ratios. Enter as a percent (20,40,60,80). Can be a list space delimited")
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
    parser.add_argument("-btr", "--begin_train", help="Year to begin training (1940-2016)", type=int)
    parser.add_argument("-etr", "--end_train", help="Year to end training. Should be higher than begin & <=2017", type=int)
    parser.add_argument("-bts", "--begin_test", help="Year to begin testing (1940-2017)", type=int)
    parser.add_argument("-ets", "--end_test", help="Year to end testing. Should be higher than begin test.", type=int)

    args = parser.parse_args()

    arguments = {}

    if args.data_path:
        arguments["data_path"] = args.data_path
    else:
        print("Default Data Path: ../Data/BWIMonthly1939.csv")
        arguments["data_path"] = "../Data/BWIMonthly1939.csv"
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

    if args.begin_train:
        begin_train = args.begin_train
    else:
        print("Default begin training is 1940")
        begin_train = 1940
    arguments["begin_train"] = begin_train

    if args.end_train:
        end_train = args.end_train
    else:
        print("Defult end training is 1980")
        end_train = 1980
    if end_train < begin_train:
        print("End_Train should be bigger than Begin_Train")
        exit(1)
    arguments["end_train"] = end_train

    if args.begin_test:
        begin_test = args.begin_test
    else:
        print("Default begin test is 1981")
        begin_test = 1981
    arguments["begin_test"] = begin_test

    if args.end_test:
        end_test = args.end_test
    else:
        print("Default end test is 2017")
        end_test = 2017
    if end_test < begin_test:
        print("End_Test should be bigger than Begin_Test")
        exit(1)
    arguments["end_test"] = end_test

    return arguments


if __name__ == "__main__":
    main()
