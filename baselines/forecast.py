############### IMPORT PACKAGES ##################

import numpy as np
from numpy.linalg import inv as inv #Used in kalman filter

#Used for naive bayes decoder
try:
    import statsmodels.api as sm
except ImportError:
    #print("\nWARNING: statsmodels is not installed. You will be unable to use the Naive Bayes Decoder")
    pass
try:
    import math
except ImportError:
    #print("\nWARNING: math is not installed. You will be unable to use the Naive Bayes Decoder")
    pass
try:
    from scipy.spatial.distance import pdist
    from scipy.spatial.distance import squareform
    from scipy.stats import norm
    from scipy.spatial.distance import cdist
except ImportError:
    #print("\nWARNING: scipy is not installed. You will be unable to use the Naive Bayes Decoder")
    pass

#Import scikit-learn (sklearn) if it is installed
try:
    from sklearn import linear_model #For Wiener Filter and Wiener Cascade
    from sklearn.svm import SVR #For support vector regression (SVR)
    from sklearn.svm import SVC #For support vector classification (SVM)
except ImportError:
    #print("\nWARNING: scikit-learn is not installed. You will be unable to use the Wiener Filter or Wiener Cascade Decoders")
    pass

#Import XGBoost if the package is installed
try:
    import xgboost as xgb #For xgboost
except ImportError:
    #print("\nWARNING: Xgboost package is not installed. You will be unable to use the xgboost decoder")
    pass

#Import functions for Keras if Keras is installed
#Note that Keras has many more built-in functions that I have not imported because I have not used them
#But if you want to modify the decoders with other functions (e.g. regularization), import them here
try:
    import keras
    keras_v1=int(keras.__version__[0])<=1
    from keras.models import Sequential
    from keras.layers import Dense, LSTM, SimpleRNN, GRU, Activation, Dropout
    from keras.utils import np_utils
except ImportError:
    #print("\nWARNING: Keras package is not installed. You will be unable to use all neural net decoders")
    pass

try:
    from sklearn.preprocessing import OneHotEncoder
except ImportError:
    #print("\nWARNING: Sklearn OneHotEncoder not installed. You will be unable to use XGBoost for Classification")
    pass

import numpy as np
from keras.models import Sequential
from keras.layers import LSTM, Dense, Dropout,Reshape




class LSTMForecasting(object):
    """
    Class for LSTM-based forecasting

    Parameters
    ----------
    units: integer, optional, default 400
        Number of hidden units in each layer

    dropout: decimal, optional, default 0
        Proportion of units that get dropped out

    num_epochs: integer, optional, default 10
        Number of epochs used for training

    verbose: binary, optional, default=0
        Whether to show progress of the fit after each epoch

    input_time_steps: integer, optional, default 10
        Number of time steps in the input sequence

    output_time_steps: integer, optional, default 5
        Number of time steps in the output sequence
    """

    def __init__(self, units=400, dropout=0, num_epochs=10, verbose=0, input_time_steps=10, output_time_steps=5):
        self.units = units
        self.dropout = dropout
        self.num_epochs = num_epochs
        self.verbose = verbose
        self.input_time_steps = input_time_steps
        self.output_time_steps = output_time_steps

    def fit(self, X_train, y_train):
        """
        Train LSTM for forecasting

        Parameters
        ----------
        X_train: numpy 3d array of shape [n_samples, input_time_steps, n_features]
            Input sequence data for training

        y_train: numpy 3d array of shape [n_samples, output_time_steps, n_features]
            Target data for training
        """

        model = Sequential()
        model.add(LSTM(self.units, input_shape=(self.input_time_steps, X_train.shape[2])))
        if self.dropout != 0:
            model.add(Dropout(self.dropout))
        #model.add(Dense( y_train.shape[1]))
        model.add(Dense(self.output_time_steps * y_train.shape[2]))
        model.add(Reshape((self.output_time_steps, y_train.shape[2])))

        model.compile(loss='mse', optimizer='rmsprop', metrics=['accuracy'])
        model.fit(X_train, y_train, epochs=self.num_epochs, verbose=self.verbose)
        self.model = model

    def predict(self, X_test):
        """
        Predict future outcomes using trained LSTM

        Parameters
        ----------
        X_test: numpy 3d array of shape [n_samples, input_time_steps, n_features]
            Input sequence data for prediction

        Returns
        -------
        y_test_predicted: numpy 3d array of shape [n_samples, output_time_steps, n_features]
            The predicted future values
        """

        y_test_predicted = self.model.predict(X_test)
        return y_test_predicted


from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Activation

class DenseNNRegression(object):

    """
    Class for the dense (fully-connected) neural network decoder

    Parameters
    ----------

    units: integer or vector of integers, optional, default 400
        This is the number of hidden units in each layer
        If you want a single layer, input an integer (e.g. units=400 will give you a single hidden layer with 400 units)
        If you want multiple layers, input a vector (e.g. units=[400,200]) will give you 2 hidden layers with 400 and 200 units, repsectively.
        The vector can either be a list or an array

    dropout: decimal, optional, default 0
        Proportion of units that get dropped out

    num_epochs: integer, optional, default 10
        Number of epochs used for training

    verbose: binary, optional, default=0
        Whether to show progress of the fit after each epoch
    """

    def __init__(self, units=400, dropout=0, num_epochs=10, verbose=0):
        self.dropout = dropout
        self.num_epochs = num_epochs
        self.verbose = verbose

        # If "units" is an integer, put it in the form of a vector
        try:
            units[0]
        except:
            units = [units]
        self.units = units

        # Determine the number of hidden layers (based on "units" that the user entered)
        self.num_layers = len(units)

    def fit(self, X_flat_train, y_train):
        """
        Train DenseNN Decoder

        Parameters
        ----------
        X_flat_train: numpy 2d array of shape [n_samples,n_features]
            This is the neural data.

        y_train: numpy 2d array of shape [n_samples, n_outputs]
            This is the outputs that are being predicted
        """

        model = Sequential()  # Declare model
        # Add first hidden layer
        model.add(Dense(self.units[0], input_dim=X_flat_train.shape[1]))  # Add dense layer
        model.add(Activation('relu'))  # Add nonlinear (relu) activation
        if self.dropout != 0:
            model.add(Dropout(self.dropout))  # Dropout some units if proportion of dropout != 0

        # Add any additional hidden layers
        for layer in range(1, self.num_layers):
            model.add(Dense(self.units[layer]))  # Add dense layer
            model.add(Activation('relu'))  # Add nonlinear (relu) activation
            if self.dropout != 0:
                model.add(Dropout(self.dropout))  # Dropout some units if proportion of dropout != 0

        # Add dense connections to all outputs
        model.add(Dense(y_train.shape[1]))  # Add final dense layer (connected to outputs)

        # Compile model (set fitting parameters)
        model.compile(loss='mse', optimizer='adam', metrics=['accuracy'])

        # Fit the model
        model.fit(X_flat_train, y_train, epochs=self.num_epochs, verbose=self.verbose)

        self.model = model

    def predict(self, X_flat_test):
        """
        Predict outcomes using trained DenseNN Decoder

        Parameters
        ----------
        X_flat_test: numpy 2d array of shape [n_samples,n_features]
            This is the neural data being used to predict outputs.

        Returns
        -------
        y_test_predicted: numpy 2d array of shape [n_samples,n_outputs]
            The predicted outputs
        """

        y_test_predicted = self.model.predict(X_flat_test)  # Make predictions
        return y_test_predicted

