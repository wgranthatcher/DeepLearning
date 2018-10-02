import pandas
import numpy as np
import matplotlib.pyplot as plt


def main():

    fourDecr_in = open("../Results/deepResults/AggregationCharts/FourDecrDifferentBackends.csv")
    fourDecr_df = pandas.read_csv(fourDecr_in)
    fourDecr_df.columns = [strip_non_ascii(x) for x in fourDecr_df.columns]

    fourDecr_tensor = fourDecr_df[fourDecr_df.backend == "tensorFlow"]
    fourDecr_tensorX = fourDecr_tensor.get("training %")
    fourDecr_tensorTime = fourDecr_tensor.get("fit_time")/60
    fourDecr_tensorAcc = fourDecr_tensor.get("test_accuracy")

    fourDecr_cntk = fourDecr_df[fourDecr_df.backend == "cntk"]
    fourDecr_cntkX = fourDecr_cntk.get("training %")
    fourDecr_cntkTime = fourDecr_cntk.get("fit_time")/60
    fourDecr_cntkAcc = fourDecr_cntk.get("test_accuracy")

    fourDecr_theano = fourDecr_df[fourDecr_df.backend == "theano"]
    fourDecr_theanoX = fourDecr_theano.get("training %")
    fourDecr_theanoTime = fourDecr_theano.get("fit_time")/60
    fourDecr_theanoAcc = fourDecr_theano.get("test_accuracy")

    oneLayer_in = open("../Results/deepResults/AggregationCharts/oneLayerDifferentBackends.csv")
    oneLayer_df = pandas.read_csv(oneLayer_in)

    oneLayer_tensor = oneLayer_df[oneLayer_df.backend == "tensorFlow"]
    oneLayer_tensorX = oneLayer_tensor.get("training %")
    oneLayer_tensorTime = oneLayer_tensor.get("fit_time")/60
    oneLayer_tensorAcc = oneLayer_tensor.get("test_accuracy")

    oneLayer_cntk = oneLayer_df[oneLayer_df.backend == "cntk"]
    oneLayer_cntkX = oneLayer_cntk.get("training %")
    oneLayer_cntkTime = oneLayer_cntk.get("fit_time")/60
    oneLayer_cntkAcc = oneLayer_cntk.get("test_accuracy")

    oneLayer_theano = oneLayer_df[oneLayer_df.backend == "theano"]
    oneLayer_theanoX = oneLayer_theano.get("training %")
    oneLayer_theanoTime = oneLayer_theano.get("fit_time")/60
    oneLayer_theanoAcc = oneLayer_theano.get("test_accuracy")

    fourSame_in = open("../Results/deepResults/AggregationCharts/FourSameDifferentBackends.csv")
    fourSame_df = pandas.read_csv(fourSame_in)

    fourSame_tensor = fourSame_df[fourSame_df.backend == "tensorFlow"]
    fourSame_tensorX = fourSame_tensor.get("training %")
    fourSame_tensorTime = fourSame_tensor.get("fit_time")/60
    fourSame_tensorAcc = fourSame_tensor.get("test_accuracy")

    fourSame_cntk = fourSame_df[fourSame_df.backend == "cntk"]
    fourSame_cntkX = fourSame_cntk.get("training %")
    fourSame_cntkTime = fourSame_cntk.get("fit_time")/60
    fourSame_cntkAcc = fourSame_cntk.get("test_accuracy")

    fourSame_theano = fourSame_df[fourSame_df.backend == "theano"]
    fourSame_theanoX = fourSame_theano.get("training %")
    fourSame_theanoTime = fourSame_theano.get("fit_time")/60
    fourSame_theanoAcc = fourSame_theano.get("test_accuracy")

    plt.figure(1)
    plt.plot(oneLayer_cntkX, oneLayer_cntkTime, linestyle='--', marker='*', color='g', linewidth=3, markersize=13,label='One Layer - CNTK')
    plt.plot(oneLayer_tensorX, oneLayer_tensorTime, marker='*', color='g', linewidth=3, markersize=13,label='One Layer - Tensor Flow')
    plt.plot(oneLayer_theanoX, oneLayer_theanoTime, linestyle=':', marker='*', color='g',linewidth=3, markersize=13, label='One Layer - Theano')

    plt.plot(fourSame_cntkX, fourSame_cntkTime, linestyle='--', marker='^', color='b',linewidth=3, markersize=13, label='Four Same - CNTK')
    plt.plot(fourSame_theanoX, fourSame_theanoTime, linestyle=':', marker='^', color='b',linewidth=3, markersize=13, label='Four Same - Theano')
    plt.plot(fourSame_tensorX, fourSame_tensorTime, marker='^', color='b', linewidth=3, markersize=13,label='Four Same - Tensor Flow')

    plt.plot(fourDecr_cntkX, fourDecr_cntkTime, linestyle='--', marker='s', color='r', linewidth=3, markersize=13,label='Four Decr - CNTK')
    plt.plot(fourDecr_tensorX, fourDecr_tensorTime, marker='s', color='r', linewidth=3, markersize=13,label='Four Decr - Tensor Flow')
    plt.plot(fourDecr_theanoX, fourDecr_theanoTime, linestyle=':', marker='s', color='r', linewidth=3, markersize=13,label='Four Decr - Theano')

    # plot formatting
    plt.legend(loc='upper left', fontsize=18)
    plt.xticks(np.arange(20, 100, 20))
    plt.xlabel('Training Ratio', fontsize=20)
    plt.ylabel('Training Time (Minutes)', fontsize=20)
    plt.tick_params(labelsize=20)
    plt.grid()

    plt.figure(2)
    plt.plot(oneLayer_cntkX, oneLayer_cntkAcc, linestyle='--', marker='*', color='g',linewidth=3, markersize=13, label='One Layer - CNTK')
    plt.plot(oneLayer_tensorX, oneLayer_tensorAcc, marker='*', color='g',linewidth=3, markersize=13, label='One Layer - Tensor Flow')
    plt.plot(oneLayer_theanoX, oneLayer_theanoAcc, linestyle=':', marker='*', linewidth=3, markersize=13,color='g', label='One Layer - Theano')

    plt.plot(fourSame_cntkX, fourSame_cntkAcc, linestyle='--', marker='^', color='b', linewidth=3, markersize=13,label='Four Same - CNTK')
    plt.plot(fourSame_tensorX, fourSame_tensorAcc, marker='^', color='b',linewidth=3, markersize=13, label='Four Same - Tensor Flow')
    plt.plot(fourSame_theanoX, fourSame_theanoAcc, linestyle=':', marker='^', color='b', linewidth=3, markersize=13,label='Four Same - Theano')

    plt.plot(fourDecr_cntkX, fourDecr_cntkAcc, linestyle='--', marker='s', color='r',linewidth=3, markersize=13, label='Four Decr - CNTK')
    plt.plot(fourDecr_tensorX, fourDecr_tensorAcc, marker='s', color='r', linewidth=3, markersize=13,label='Four Decr - Tensor Flow')
    plt.plot(fourDecr_theanoX, fourDecr_theanoAcc, linestyle=':', marker='s', color='r', linewidth=3, markersize=13,label='Four Decr - Theano')

    # plot formatting
    plt.legend(loc='lower right', fontsize=18)
    plt.xticks(np.arange(20, 100, 20))
    plt.xlabel('Training Ratio', fontsize=20)
    plt.ylabel('Accuracy', fontsize=20)
    plt.tick_params(labelsize=20)
    plt.grid()
    plt.show()


def strip_non_ascii(string):
    ''' Returns the string without non ASCII characters'''
    stripped = (c for c in string if 0 < ord(c) < 127)
    return ''.join(stripped)


if __name__=="__main__":
    main()
