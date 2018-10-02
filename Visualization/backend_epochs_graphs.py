import pandas
import numpy as np
import matplotlib.pyplot as plt


def main():
    oneLayer_in = open("../Results/deepResults/AggregationCharts/oneLayerEpochs.csv")
    oneLayer_df = pandas.read_csv(oneLayer_in)
    oneLayer_in.close()
    print(oneLayer_df)
    oneLayer_df.columns = [strip_non_ascii(x) for x in oneLayer_df.columns]

    oneLayer_cntk = oneLayer_df[oneLayer_df.backend == "cntk"]
    oneLayer_cntkX = oneLayer_cntk.get("param_epochs")
    oneLayer_cntkTime = oneLayer_cntk.get("mean_fit_time") / 60
    oneLayer_cntkAcc = oneLayer_cntk.get("mean_test_score")

    oneLayer_theano = oneLayer_df[oneLayer_df.backend == "theano"]
    oneLayer_theanoX = oneLayer_theano.get("param_epochs")
    oneLayer_theanoTime = oneLayer_theano.get("mean_fit_time") / 60
    oneLayer_theanoAcc = oneLayer_theano.get("mean_test_score")

    oneLayer_tensor = oneLayer_df[oneLayer_df.backend == "tensorFlow"]
    oneLayer_tensorX = oneLayer_tensor.get("param_epochs")
    oneLayer_tensorTime = oneLayer_tensor.get("mean_fit_time") / 60
    oneLayer_tensorAcc = oneLayer_tensor.get("mean_test_score")

    fourSame_in = open("../Results/deepResults/AggregationCharts/fourSameEpochs.csv")
    fourSame_df = pandas.read_csv(fourSame_in)
    fourSame_in.close()
    fourSame_df.columns = [strip_non_ascii(x) for x in fourSame_df.columns]

    fourSame_cntk = fourSame_df[fourSame_df.backend == "cntk"]
    fourSame_cntkX = fourSame_cntk.get("param_epochs")
    fourSame_cntkTime = fourSame_cntk.get("mean_fit_time") / 60
    fourSame_cntkAcc = fourSame_cntk.get("mean_test_score")

    fourSame_theano = fourSame_df[fourSame_df.backend == "theano"]
    fourSame_theanoX = fourSame_theano.get("param_epochs")
    fourSame_theanoTime = fourSame_theano.get("mean_fit_time") / 60
    fourSame_theanoAcc = fourSame_theano.get("mean_test_score")

    fourSame_tensor = fourSame_df[fourSame_df.backend == "tensorFlow"]
    fourSame_tensorX = fourSame_tensor.get("param_epochs")
    fourSame_tensorTime = fourSame_tensor.get("mean_fit_time") / 60
    fourSame_tensorAcc = fourSame_tensor.get("mean_test_score")

    fourDecr_in = open("../Results/deepResults/AggregationCharts/fourDecrEpochs.csv")
    fourDecr_df = pandas.read_csv(fourDecr_in)
    fourDecr_in.close()
    fourDecr_df.columns = [strip_non_ascii(x) for x in fourDecr_df.columns]

    fourDecr_cntk = fourDecr_df[fourDecr_df.backend == "cntk"]
    fourDecr_cntkX = fourDecr_cntk.get("param_epochs")
    fourDecr_cntkTime = fourDecr_cntk.get("mean_fit_time") / 60
    fourDecr_cntkAcc = fourDecr_cntk.get("mean_test_score")

    fourDecr_theano = fourDecr_df[fourDecr_df.backend == "theano"]
    fourDecr_theanoX = fourDecr_theano.get("param_epochs")
    fourDecr_theanoTime = fourDecr_theano.get("mean_fit_time") / 60
    fourDecr_theanoAcc = fourDecr_theano.get("mean_test_score")

    fourDecr_tensor = fourDecr_df[fourDecr_df.backend == "tensorFlow"]
    fourDecr_tensorX = fourDecr_tensor.get("param_epochs")
    fourDecr_tensorTime = fourDecr_tensor.get("mean_fit_time") / 60
    fourDecr_tensorAcc = fourDecr_tensor.get("mean_test_score")

    plt.figure(1)
    plt.plot(oneLayer_cntkX, oneLayer_cntkTime, linestyle='--', marker='*',  color='g', linewidth=3, markersize=13, label='One Layer - CNTK')
    plt.plot(oneLayer_tensorX, oneLayer_tensorTime, marker='*',linewidth=3, color='g', markersize=13,label='One Layer - Tensor Flow')
    plt.plot(oneLayer_theanoX, oneLayer_theanoTime, linestyle=':',linewidth=3, marker='*',markersize=13, color='g', label='One Layer - Theano')

    plt.plot(fourSame_cntkX, fourSame_cntkTime, linestyle='--',linewidth=3, marker='^', markersize=13,color='b', label='Four Same - CNTK')
    plt.plot(fourSame_tensorX, fourSame_tensorTime, linestyle=':', linewidth=3,marker='^', markersize=13,color='b', label='Four Same - Tensor Flow')
    plt.plot(fourSame_theanoX, fourSame_theanoTime, linestyle=':', linewidth=3,marker='^',markersize=13, color='b', label='Four Same - Theano')

    plt.plot(fourDecr_cntkX, fourDecr_cntkTime, linestyle='--',linewidth=3, marker='s',markersize=13, color='r', label='Four Decr - CNTK')
    plt.plot(fourDecr_tensorX, fourDecr_tensorTime, marker='s', linewidth=3,color='r', markersize=13,label='Four Decr - Tensor Flow')
    plt.plot(fourDecr_theanoX, fourDecr_theanoTime, linestyle=':',linewidth=3, marker='s',markersize=13, color='r', label='Four Decr - Theano')

    # plot formatting
    plt.legend(loc='upper left', fontsize=18)
    plt.xticks(np.arange(0, 18, 2))
    plt.xlabel('Epochs', fontsize=20)
    plt.ylabel('Training Time (Minutes)', fontsize=20)
    plt.tick_params(labelsize=20)
    plt.grid()

    plt.figure(2)
    plt.plot(oneLayer_cntkX, oneLayer_cntkAcc, linestyle='--', linewidth=3, marker='*',markersize=13, color='g', label='One Layer - CNTK')
    plt.plot(oneLayer_tensorX, oneLayer_tensorAcc, marker='*', linewidth=3,color='g', markersize=13,label='One Layer - Tensor Flow')
    plt.plot(oneLayer_theanoX, oneLayer_theanoAcc, linestyle=':',linewidth=3, marker='*',markersize=13, color='g', label='One Layer - Theano')

    plt.plot(fourSame_cntkX, fourSame_cntkAcc, linestyle='--', linewidth=3,marker='^',markersize=13, color='b', label='Four Same - CNTK')
    plt.plot(fourSame_tensorX, fourSame_tensorAcc, marker='^', linewidth=3,color='b',markersize=13, label='Four Same - Tensor Flow')
    plt.plot(fourSame_theanoX, fourSame_theanoAcc, linestyle=':', linewidth=3,marker='^', markersize=13,color='b', label='Four Same - Theano')

    plt.plot(fourDecr_cntkX, fourDecr_cntkAcc, linestyle='--',linewidth=3, marker='s',markersize=13, color='r', label='Four Decr - CNTK')
    plt.plot(fourDecr_tensorX, fourDecr_tensorAcc, marker='s', linewidth=3,color='r',markersize=13, label='Four Decr - Tensor Flow')
    plt.plot(fourDecr_theanoX, fourDecr_theanoAcc, linestyle=':',linewidth=3, marker='s',markersize=13, color='r', label='Four Decr - Theano')

    # plot formatting
    plt.legend(loc='lower right', fontsize=18)
    plt.xticks(np.arange(0, 18, 2))
    plt.xlabel('Epochs', fontsize=20)
    plt.ylabel('Accuracy', fontsize=20)
    plt.ylim([0.91,0.945])
    plt.tick_params(labelsize=20)
    plt.grid()

    plt.show()


def strip_non_ascii(string):
    ''' Returns the string without non ASCII characters'''
    stripped = (c for c in string if 0 < ord(c) < 127)
    return ''.join(stripped)


if __name__=="__main__":
    main()