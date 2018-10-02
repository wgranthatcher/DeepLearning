import pandas
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from mpl_toolkits.mplot3d import axes3d
from matplotlib import cm
from pylab import meshgrid
import numpy as np

def main():
    data_dir = '/home/josh/Documents/COSC/Research/APK_project/DeepLearningResearch/Results/shallowResults/'
    perms_data = data_dir+'DecisionTree_permissions_01-31-2252.csv'
    features_data = data_dir+'DecisionTree_features_01-31-2254.csv'
    combined_data = data_dir+'DecisionTree_combined_01-31-2245.csv'
    perms_in = open(perms_data)
    features_in = open(features_data)
    combined_in = open(combined_data)

    perms_df = pandas.read_csv(perms_in, index_col=0)
    features_df = pandas.read_csv(features_in, index_col=0)
    combined_df = pandas.read_csv(combined_in, index_col=0)

    ratios = [0.2, 0.4, 0.6, 0.8]

    #perms x and y
    perms_ratios = perms_df.index.tolist()
    perms_acc = perms_df.get("f1_score").values

    #features x and y
    features_ratios = features_df.index.tolist()
    features_acc = features_df.get("f1_score").values

    #combined x and y
    combined_ratios = combined_df.index.tolist()
    combined_acc = combined_df.get("f1_score").values

    #create line for each dataset
    permissions_only = plt.plot(ratios, perms_acc, marker='*', markersize=12, label='Permissions only')
    features_only = plt.plot(ratios, features_acc, marker='o', markersize=12, label='Features only')
    combined_data = plt.plot(ratios, combined_acc, marker='v', markersize=12, label='Combined data')

    #plot formatting
    plt.title('f1_score vs. input data source', fontsize=18)
    plt.legend(loc='upper left', fontsize=12)
    plt.ylim=([2, 3.2])
    plt.xlabel('Training Ratio', fontsize=18)
    plt.ylabel('F1_score', fontsize=18)
    plt.tick_params(labelsize=14)

    plt.show()

if __name__=="__main__":
    main()
