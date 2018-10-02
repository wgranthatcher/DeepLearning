import pandas
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import axes3d
from matplotlib import cm
from pylab import meshgrid
import numpy as np

def main():
    csv_in = open("dropoutWeight_oneLayer/weightedDropoutoneLayer.csv", 'r')
    df = pandas.read_csv(csv_in)

    weight = df.get("param_weight_constraint").as_matrix()
    dropout_rate = df.get("param_dropout_rate").as_matrix()
    score = df.get("mean_test_score").as_matrix()

    fig = plt.figure()
    ax = fig.gca(projection='3d')
    ax.set_title('Grid Search for Weight Constraint and Dropout Rate', fontsize=28, fontweight='bold')

    surf = ax.plot_trisurf(weight, dropout_rate, score, cmap=cm.coolwarm, linewidth=0)

    ax.set_xlim(0,6)
    ax.set_ylim(0,.6)
    ax.set_zlim(0.88, .95)
    ax.set_xlabel("Weight Constraint", fontsize=22 ,labelpad=14)
    ax.set_ylabel("Dropout Rate", fontsize=22, labelpad=14)
    ax.set_zlabel("Accuracy", fontsize=22, labelpad=14)
    ax.tick_params(axis='x', labelsize=16)
    ax.tick_params(axis='y', labelsize=16)
    ax.tick_params(axis='z', labelsize=16)

    plt.show()

if __name__=="__main__":
    main()
