import pandas
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import axes3d
from matplotlib import cm
from pylab import meshgrid
import numpy as np

def main():
    csv_in = open("/home/josh/Documents/COSC/research/ml_malware/DeepLearningResearch/Results/deepResults/25OctTestMaster/epochBatchTrainoneLayer.csv", 'r')
    df = pandas.read_csv(csv_in)

    batch = df.get("param_batch_size").as_matrix()
    epoch = df.get("param_epochs").as_matrix()
    score = df.get("mean_test_score").as_matrix()

    fig = plt.figure()
    ax = fig.gca(projection='3d')
    ax.set_title('Grid Search for Batch Size and Epochs', fontsize=20, fontweight='bold')

    surf = ax.plot_trisurf(batch, epoch, score, cmap=cm.coolwarm, linewidth=0)

    ax.set_xlim(5000,0)
    ax.set_ylim(0,32)
    ax.set_zlim(0.6, 1.0)
    ax.set_xlabel("batch size", fontsize=16 ,labelpad=10)
    ax.set_ylabel("epochs", fontsize=16, labelpad=10)
    ax.set_zlabel("accuracy", fontsize=16, labelpad=10)
    ax.tick_params(axis='x', labelsize=14)
    ax.tick_params(axis='y', labelsize=14)
    ax.tick_params(axis='z', labelsize=14)

    plt.show()






'''
    Z = X+Y
    fig = plt.figure()
    ax = fig.gca(projection='3d')

    surf = ax.plot_surface(X, Y, Z, cmap=cm.coolwarm, linewidth=0)

    ax.set_ylim(0,10)
    ax.set_zlim(0,20)
    ax.set_xlim(0,10)

    plt.show()
'''





if __name__=="__main__":
    main()
