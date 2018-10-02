import pandas
import matplotlib.pyplot as plt


def main():
    nadam_in = open("neuronsOptimoneLayer/neurons_acc_nadam.csv", 'r')
    adadelta_in = open("neuronsOptimoneLayer/neuronsOptimoneLayer_adadelta.csv", 'r')
    adamax_in = open("neuronsOptimoneLayer/neuronsOptimoneLayer_adamax.csv", 'r')
    adam_in = open("neuronsOptimoneLayer/neuronsOptimoneLayer_adam.csv", 'r')
    RMSprop_in = open("neuronsOptimoneLayer/neuronsOptimoneLayer_RMSprop.csv", 'r')
    SGD_in = open("neuronsOptimoneLayer/neuronsOptimoneLayer_SGD.csv", 'r')

    nadam_df = pandas.read_csv(nadam_in)
    adadelta_df = pandas.read_csv(adadelta_in)
    adamax_df = pandas.read_csv(adamax_in)
    adam_df = pandas.read_csv(adam_in)
    RMS_df = pandas.read_csv(RMSprop_in)
    SGD_df = pandas.read_csv(SGD_in)

    #nadam x and y
    nadam_neurons = nadam_df.get("param_neurons")
    nadam_score = nadam_df.get("mean_test_score")

    #adadelta x and y
    adadelta_neurons = adadelta_df.get("param_neurons")
    adadelta_score = adadelta_df.get("mean_test_score")

    #adamax x and y
    adamax_neurons = adamax_df.get("param_neurons")
    adamax_score = adamax_df.get("mean_test_score")

    #adam x and y
    adam_neurons = adam_df.get("param_neurons")
    adam_score = adam_df.get("mean_test_score")

    #RMS x and y
    RMS_neurons = RMS_df.get("param_neurons")
    RMS_score = RMS_df.get("mean_test_score")

    #SGD x and y
    SGD_neurons = SGD_df.get("param_neurons")
    SGD_score = SGD_df.get("mean_test_score")

    plt.figure(1)
    #create line for each optimizer
    plt.plot(adadelta_neurons, adadelta_score, marker='*', linewidth=3, markersize=13,label='adadelta')
    plt.plot(adamax_neurons, adamax_score ,marker='o', linewidth=3, markersize=13,label='adamax')
    plt.plot(adam_neurons, adam_score, marker='v',linewidth=3, markersize=13, label='adam')
    plt.plot(nadam_neurons, nadam_score, marker='^', linewidth=3, markersize=13,label='nadam')
    plt.plot(RMS_neurons, RMS_score, marker='s',linewidth=3, markersize=13, label='RMSprop')
    plt.plot(SGD_neurons, SGD_score, marker='p',linewidth=3, markersize=13, label='SGD')

    #plot formatting
    plt.legend(loc='upper left', fontsize = 18)
    #plt.ylim([.92, .95])
    #plt.title('One Layer Model - Optimizer and Neurons', fontsize=28)
    plt.xlabel('Neurons', fontsize = 20)
    plt.ylabel('Accuracy', fontsize = 20)
    plt.tick_params(labelsize=20)
    plt.grid()
    #plt.legend([adadelta, adamax, adam, RMSprop, SGD]) #['adadelta', 'adamax', 'adam','nadam', 'RMSprop,', 'SGD'])


    plt.show()




if __name__=="__main__":
    main()
