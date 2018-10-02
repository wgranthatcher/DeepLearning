from keras.utils import plot_model
from keras.wrappers.scikit_learn import KerasClassifier
from sklearn.model_selection import StratifiedShuffleSplit, cross_validate, GridSearchCV
from sklearn.feature_extraction.text import CountVectorizer

import sys


def main():
    sys.path.insert(0, 'C:/Users/jarre/OneDrive/Documents/Research/Fall2017Git/Classification')
    import keras_models
    epochs = 16
    batch_size = 10
    neurons = 50
    optimizer = 'nadam'
    weight_constraint = 5
    dropout_rate = .10

    model = keras_models.create_one_layer()
    plot_model(model, to_file='./figures/oneLayer.png', show_shapes=True)
    # model1 = keras_models.create_fourSameLayer()
    # plot_model(model1, to_file='./figures/fourSame.png', show_shapes=True)
    # model2 = keras_models.create_fourDecrLayer()
    # plot_model(model2, to_file='./figures/fourDecr.png', show_shapes=True)


if __name__ == "__main__":
    main()