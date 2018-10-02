import pandas
import os
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from mpl_toolkits.mplot3d import axes3d
from matplotlib import cm
from pylab import meshgrid
import numpy as np


def main():

    base_path = os.path.abspath('../..')
    results_path = os.path.join(base_path, 'Results/deepResults/multi_input/final_test')
    print 'base path: ' + base_path
    print 'results_path: ' + results_path

     #= open("neuronsOptimoneLayer/neurons_acc_nadam.csv", 'r')
    dual_simple_in = open(os.path.join(results_path, 'dual_simple0327-0843.csv'), 'r')
    dual_large_in = open(os.path.join(results_path, 'dual_large0327-0137.csv'), 'r')
    oneLayer_feat_in = open(os.path.join(results_path, 'oneLayer_feat0326-1756.csv'), 'r')
    oneLayer_perm_in = open(os.path.join(results_path, 'oneLayer_perm0326-1634.csv'), 'r')
    oneLayer_comb_in = open(os.path.join(results_path, 'oneLayer_comb0326-2315.csv'), 'r')

    dual_simple_df = pandas.read_csv(dual_simple_in)
    dual_large_df = pandas.read_csv(dual_large_in)
    oneLayer_feat_df = pandas.read_csv(oneLayer_feat_in)
    oneLayer_perm_df = pandas.read_csv(oneLayer_perm_in)
    oneLayer_comb_df = pandas.read_csv(oneLayer_comb_in)


    #dual_simple x and y
    ds_ratio = dual_simple_df.get("train_ratio")
    ds_f1 = dual_simple_df.get("f1_score")

    #dual_large x and y
    dl_ratio = dual_large_df.get("train_ratio")
    dl_f1 = dual_large_df.get("f1_score")

    #oneLayer_feat x and y
    ol_feat_ratio = oneLayer_feat_df.get("train_ratio")
    ol_feat_f1 = oneLayer_feat_df.get("f1_score")

    #oneLayer_perm x and y
    ol_perm_ratio = oneLayer_perm_df.get("train_ratio")
    ol_perm_f1 = oneLayer_perm_df.get("f1_score")

    #oneLayer_comb x and y
    ol_comb_ratio = oneLayer_perm_df.get("train_ratio")
    ol_comb_f1 = oneLayer_perm_df.get("f1_score")

    #create line for each optimizer
    dual_simple = plt.plot(ds_ratio, ds_f1, marker='*', markersize=16, label='dual_simple')
    dual_large = plt.plot(dl_ratio, dl_f1, marker='o', markersize=16, label='dual_large')
    ol_feat = plt.plot(ol_feat_ratio, ol_feat_f1, marker='v', markersize=16, label='one_layer_features')
    ol_perm = plt.plot(ol_perm_ratio, ol_perm_f1, marker='^', markersize=16, label='one_layer_permissions')
    ol_comb = plt.plot(ol_comb_ratio, ol_comb_f1, marker='s', markersize=16, label='one_layer_combined')

    #adam = plt.plot(adam_neurons, adam_score, marker='v', markersize=16, label='adam')
    #nadam = plt.plot(nadam_neurons, nadam_score, marker='^', markersize=16, label='nadam')
    #RMSprop = plt.plot(RMS_neurons, RMS_score, marker='s', markersize=16, label='RMSprop')
    #SGD = plt.plot(SGD_neurons, SGD_score, marker='p', markersize=16, label='SGD')

    #plot formatting
    plt.legend(loc='upper left', fontsize = 16)
    plt.ylim([.65,1])
    plt.title('Model F1 Score and Training Ratio', fontsize=28)
    plt.xlabel('Training Ratio', fontsize = 26)
    plt.ylabel('F1 Score', fontsize = 26)
    plt.tick_params(labelsize=20)
    #plt.legend([adadelta, adamax, adam, RMSprop, SGD]) #['adadelta', 'adamax', 'adam','nadam', 'RMSprop,', 'SGD'])


    plt.show()


if __name__ == "__main__":
    main()
