import sys
from operator import truediv
from axolotl import *
import numpy as np
import matplotlib.pyplot as plt
from keras.models import Sequential
from keras.layers import Dense
from sklearn.model_selection import train_test_split
from keras import backend as K

# config
print_predictions = True
graph_predictions = True
graph_error_dist = True

ANDROID_HEIGHT = 2280. / 444  # 1920px at 401ppi per https://www.apple.com/iphone-7/specs/
ANDROID_WIDTH = 1080. / 444  # 1080px at 401ppi


def in_distance(y_true, y_pred):
    y_error = y_true - y_pred
    y_error_normalized = (y_error) / 2  # the width is currently 2 (as the coordinates are [-1, 1])
    y_scaled_error = K.dot(y_error_normalized, K.constant(np.array([[ANDROID_WIDTH, 0], [0, ANDROID_HEIGHT]])))
    y_distance_sq = K.sum(K.square(y_scaled_error), axis=-1)
    y_distance = K.sqrt(y_distance_sq)
    return y_distance


def in_dist_mean(*args, **kwargs):
    return K.mean(in_distance(*args, **kwargs))


def train_location_model(data):
    # find windows where touching
    touching_windows, touching_labels = get_touching_windows(data, with_labels=True)
    expanded_touching_windows = expand_windows_interpolated(data, touching_windows)
    # convert to feature vectors
    positive_feature_vectors = feature_vectors_from_windows(expanded_touching_windows)
    # split into input (X) and output (Y) variables
    X = np.array(map(np.array, positive_feature_vectors))
    Y = np.array(map(np.array, touching_labels))
    # create model
    model = Sequential()
    model.add(Dense(window_samples * 4, input_dim=window_samples * 6, activation='linear'))
    model.add(Dense(window_samples * 2, activation='linear'))
    model.add(Dense(window_samples, activation='linear'))
    model.add(Dense(2, activation='linear'))
    # Compile model
    model.compile(loss='mse', optimizer='adam', metrics=['mse', in_dist_mean])
    # Fit the model
    model.fit(X, Y, nb_epoch=40, batch_size=20, verbose=0)
    return model


def learn_location(accel_file, gyro_file, verbose=True):
    # read the data in
    data = read_data(accel_file, gyro_file)

    # find windows where touching
    touching_windows, touching_labels = get_touching_windows(data, with_labels=True)
    expanded_touching_windows = expand_windows_interpolated(data, touching_windows)

    # convert to feature vectors
    positive_feature_vectors = feature_vectors_from_windows(expanded_touching_windows)

    # learn

    # fix random seed for reproducibility
    seed = 12
    np.random.seed(seed)

    # split into input (X) and output (Y) variables
    X = np.array(map(np.array, positive_feature_vectors))
    Y = np.array(map(np.array, touching_labels))

    # X = np.load(accel_file.replace("accel.txt", "X.npy"))
    # Y = np.load(accel_file.replace("accel.txt", "Y.npy"))

    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.33)

    # create model
    model = Sequential()
    model.add(Dense(window_samples * 4, input_dim=window_samples * 6, activation='linear'))
    model.add(Dense(window_samples * 2, activation='linear'))
    model.add(Dense(window_samples, activation='linear'))
    model.add(Dense(2, activation='linear'))

    # Compile model
    model.compile(loss='mse', optimizer='adam', metrics=['mse', in_dist_mean])

    # Fit the model
    verbosity = 0
    if verbose:
        verbosity = 1
    model.fit(X_train, Y_train, validation_data=(X_test, Y_test), nb_epoch=500, batch_size=30, verbose=verbosity)

    # print predictions
    if print_predictions and verbose:
        for x_val, y_val in zip(X_test, Y_test):
            print (y_val, ":", model.predict(np.array([x_val]), verbose=0))

    # Show histogram of data
    if graph_error_dist:
        pred = model.predict(X_test)
        # plt.hist(K.eval(in_distance(K.constant(Y_test), K.constant(pred))), bins=20, normed=True)
        # plt.show()

    # graph predictions
    if graph_predictions and verbose:
        # m = 34
        # m = 1
        # n = 3187
        m = 22
        n = 10

        pred_data = zip(X_test, Y_test)

        accuracy = []
        distanceError = []
        for ii in xrange(min(m * n, len(X))):
            x_val, y_val = pred_data[ii]

            curr_plot = plt.subplot(m, n, ii + 1)  # the position parameter is 1-indexed

            plt.plot(y_val[0], -1 * y_val[1], 'go')
            pred = model.predict(np.array([x_val]), verbose=0).flatten()
            plt.plot(pred[0], -1 * pred[1], 'ro')

            if coordinatesToDigit(deNormX(y_val[0]), deNormY(y_val[1])) != "Not Numeric":
                print("Coordinates: ", deNormX(y_val[0]), deNormY(y_val[1]), "|", deNormX(pred[0]), deNormY(pred[1]))
                print("Button: ", coordinatesToDigit(deNormX(y_val[0]), deNormY(y_val[1])), "|",
                      coordinatesToDigit(deNormX(pred[0]), deNormY(pred[1])))
                print

                accuracy.append((coordinatesToDigit(deNormX(y_val[0]), deNormY(y_val[1])),
                                 coordinatesToDigit(deNormX(pred[0]), deNormY(pred[1]))))

            curr_plot.xaxis.set_visible(False)
            curr_plot.yaxis.set_visible(False)

            plt.ylim(-1, 1)
            plt.xlim(-1, 1)

        # plt.show()

    correctPredictions = 0
    touch_isNumber = 0
    for item in accuracy:
        if item[0] != "Not Numeric":
            touch_isNumber += 1
            if item[0] == item[1]:
                correctPredictions += 1

    correctPredictionsPercentage = float(truediv(correctPredictions, touch_isNumber)) * 100
    print "Correct Predictions: " + str(correctPredictions) + "/" + str(touch_isNumber) + " | " + str(
        "{:.2f}".format(correctPredictionsPercentage)) + "%"
    print

    return correctPredictionsPercentage
    return dict(zip(model.metrics_names, model.evaluate(X_test, Y_test, verbose=verbosity)))


def deNormX(normVal):
    denormVal = (normVal + 1) * 1080 / 2
    return denormVal


def deNormY(normVal):
    denormVal = (normVal + 1) * 2280 / 2
    return denormVal


def coordinatesToDigit(x, y):
    button = "Not Numeric"
    x = int(x)
    y = int(y)
    if x in range(0, 272) and y in range(1490, 1650):
        button = "1"
    elif x in range(273, 538) and y in range(1490, 1650):
        button = "2"
    elif x in range(545, 804) and y in range(1490, 1650):
        button = "3"
    elif x in range(0, 272) and y in range(1655, 1814):
        button = "4"
    elif x in range(273, 538) and y in range(1655, 1814):
        button = "5"
    elif x in range(545, 804) and y in range(1655, 1814):
        button = "6"
    elif x in range(0, 272) and y in range(1820, 1981):
        button = "7"
    elif x in range(273, 538) and y in range(1820, 1981):
        button = "8"
    elif x in range(545, 804) and y in range(1820, 1981):
        button = "9"
    elif x in range(278, 538) and y in range(1985, 2151):
        button = "0"
    return button


if __name__ == "__main__":
    # 2Hands dataset, Best result: 41.84%, nb_epoch=500, batch_size=30
    learn_location("diamantTest_accel_2hands.txt", "diamantTest_gyro_2hands.txt")
    # learn_location("1handtyping1hour_accel.txt", "1handtyping1hour_gyro.txt")
    # 1Hand dataset,Best result: 29.49%, nb_epoch=50,batch_size=10
    # learn_location("diamantTest_accel_1hand.txt", "diamantTest_gyro_1hand.txt")
