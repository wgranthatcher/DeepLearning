import pandas
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import axes3d
from matplotlib import cm
from pylab import meshgrid
import numpy as np

def main():
    fourSame_in = open("finalCV/finalCV_fourSame.csv", 'r')
    fourDec_in = open("finalCV/finalCV_fourDec.csv", 'r')
    oneLayer_in = open("finalCV/finalCV_oneLayer.csv", 'r')

    fourSame_df = pandas.read_csv(fourSame_in)
    fourDec_df = pandas.read_csv(fourDec_in)
    oneLayer_df = pandas.read_csv(oneLayer_in)

    #fourSame x and y
    fourSame_ratios = fourSame_df.get("Ratio")
    fourSame_acc = fourSame_df.get("test_f1")

    #fourDec x and y
    fourDec_ratios = fourDec_df.get("Ratio")
    fourDec_acc = fourDec_df.get("test_f1")

    #oneLayer x and y
    oneLayer_ratios = oneLayer_df.get("Ratio")
    oneLayer_acc = oneLayer_df.get("test_f1")

    fourSame = plt.plot(fourSame_ratios, fourSame_acc, marker='s', color='b',linewidth=3, markersize=13, label='Four Same Size')
    fourDec = plt.plot(fourDec_ratios, fourDec_acc, marker='*', color='r', linewidth=3, markersize=13,label='Four Decreasing Size')
    oneLayer = plt.plot(oneLayer_ratios, oneLayer_acc, marker='^', color='g', linewidth=3, markersize=13,label='One Layer')

    #plot formatting
    plt.legend(loc='upper left', fontsize = 18)
    #plt.ylim([.90,1])
    plt.xlabel('Training Ratio', fontsize = 20)
    plt.ylabel('F1 score', fontsize = 20)
    plt.tick_params(labelsize=20)
    plt.grid()

    plt.show()

if __name__=="__main__":
    main()
