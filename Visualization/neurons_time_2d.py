import pandas
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import axes3d
from matplotlib import cm
from pylab import meshgrid
import numpy as np

def main():
    fourSame_in = open("neuronsOptim_time/neuronsOptim_time_fourDec.csv", 'r')
    fourDec_in = open("neuronsOptim_time/neuronsOptim_time_fourSame.csv", 'r')
    oneLayer_in = open("neuronsOptim_time/neuronsOptim_time_oneLayer.csv", 'r')

    fourSame_df = pandas.read_csv(fourSame_in)
    fourDec_df = pandas.read_csv(fourDec_in)
    oneLayer_df = pandas.read_csv(oneLayer_in)

    #fourSame x and y
    fourSame_neurons = fourSame_df.get("Neurons")
    fourSame_time = fourSame_df.get("Time")

    #fourDec x and y
    fourDec_neurons = fourDec_df.get("Neurons")
    fourDec_time = fourDec_df.get("Time")

    #oneLayer x and y
    oneLayer_neurons = oneLayer_df.get("Neurons")
    oneLayer_time = oneLayer_df.get("Time")

    fourSame = plt.plot(fourSame_neurons, fourSame_time, marker='s', color='b',linewidth=3, markersize=13, label='Four Same Size')
    fourDec = plt.plot(fourDec_neurons, fourDec_time, marker='*', color='r', linewidth=3, markersize=13,label='Four Decreasing Size')
    oneLayer = plt.plot(oneLayer_neurons, oneLayer_time, marker='^', color='g', linewidth=3, markersize=13,label='One Layer')

    #plot formatting
    plt.legend(loc='upper left', fontsize = 18)
    plt.ylim([0,500])
    #plt.title('Number of Neurons and Training Time', fontsize = 28)
    plt.xlabel('Neurons', fontsize = 20)
    plt.ylabel('Time (seconds)', fontsize = 20)
    plt.tick_params(labelsize=20)
    plt.grid()

    plt.show()




if __name__=="__main__":
    main()
