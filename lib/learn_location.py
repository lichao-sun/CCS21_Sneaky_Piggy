import sys
from operator import truediv
from axolotl import *
import numpy as np
import matplotlib.pyplot as plt
from keras.models import load_model
from keras.models import Sequential
from keras.layers import Dense, GRU, Embedding, Dropout, LSTM
from sklearn.model_selection import train_test_split
from keras import backend as K
from sklearn.metrics import accuracy_score
from keras.callbacks import ModelCheckpoint
import argparse
import math
import random

# config
print_predictions = True
graph_predictions = True
graph_error_dist = True

ANDROID_HEIGHT = 2280. / 444 # 1920px at 401ppi per https://www.apple.com/iphone-7/specs/
ANDROID_WIDTH = 1080. / 444 # 1080px at 401ppi


def in_distance(y_true, y_pred):
    y_error = y_true - y_pred
    y_error_normalized = (y_error) / 2 # the width is currently 2 (as the coordinates are [-1, 1])
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
    model.fit(X, Y, nb_epoch=params['nb_epoch'], batch_size=params['batch_size'], verbose=0)
    return model


def data_transform(X, Y, one_hot=False, digit_only=False):
    X_new = []
    Y_new = []
    if one_hot == True:
        for i in range(len(X)):
            y_new = coordinatesToDigit(deNormX(Y[i][0]), deNormY(Y[i][1]))
            if y_new == 'Not Numeric':
                y_new = '10'
            # print(y_new)
            if digit_only==True and y_new == '10':
                    continue
            else:
                X_new.append(X[i])
                Y_new.append(int(y_new))
        onehot_encoded = []
        for value in Y_new:
            letter = [0 for _ in range(11)]
            letter[value] = 1
            onehot_encoded.append(letter)
        Y = np.array(onehot_encoded)
        X = np.array(X_new)
    else:
        for i in range(len(X)):
            y_new = coordinatesToDigit(deNormX(Y[i][0]), deNormY(Y[i][1]))
            # print(y_new)
            if digit_only==True and y_new == 'Not Numeric':
                    continue
            else:
                X_new.append(X[i])
                Y_new.append(Y[i])
        Y = np.array(Y_new)
        X = np.array(X_new)
    return X, Y


def learn_location_gru_softmax(accel_file, gyro_file, params, verbose=True):
    seed = 12
    np.random.seed(seed)

    ori_accel_file = accel_file

    if params['test_experiment'] == True:
        accel_file = "/home/sunlichao/Documents/lehigh_cloud/CCS21/data_test/sample_" + params["test_file_id"] + "/accel.txt"
        gyro_file = "/home/sunlichao/Documents/lehigh_cloud/CCS21/data_test/sample_" + params["test_file_id"] + "/gyro.txt"

    if params['first_data_load'] == True:
        # read the data in
        data = read_data(accel_file, gyro_file)

        # find windows where touching
        touching_windows, touching_labels = get_touching_windows(data, with_labels=True)
        expanded_touching_windows = expand_windows_interpolated(data, touching_windows)

        # convert to feature vectors
        positive_feature_vectors = feature_vectors_from_windows(expanded_touching_windows)

        # fix random seed for reproducibility

        # split into input (X) and output (Y) variables
        X = np.array(map(np.array, positive_feature_vectors))
        Y = np.array(map(np.array, touching_labels))
        np.save(accel_file.replace("_accel", "").replace(".txt", "_X"), X)
        np.save(accel_file.replace("_accel", "").replace(".txt", "_Y"), Y)

    else:  # 1handtyping1hour_accel
        X = np.load(accel_file.replace("_accel", "").replace(".txt", "_X.npy"))
        Y = np.load(accel_file.replace("_accel", "").replace(".txt", "_Y.npy"))

    X, Y = data_transform(X, Y, True, params['digital_only'])
    print("X Y shapes: ", X.shape, Y.shape)

    print("shapes: ", X[0])

    X = X.reshape(len(X), params['data_shape'], len(X[0]) / params['data_shape'])

    if params['test_experiment'] == True:
        X_train, X_test, Y_train, Y_test = [], X, [], Y
    else:
        X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=params['test_size'])

    print("X test shapes: ", X_test.shape)

    if params['sampled'] > 0:
        X_train, Y_train = X_train[:int(len(X_train) * params['sampled'] / 60)], Y_train[:int(
            len(X_train) * params['sampled'] / 60)]
        print("XXX minutes X Y shapes: ", params['sampled'], X_train.shape, Y_train.shape)

    Y_test_eval = np.array([np.where(r==1)[0][0] for r in Y_test])

    # create model
    #X = X.reshape(len(X), params['data_shape'], len(X[0]) / params['data_shape'])
    model = Sequential()
    # Add a LSTM layer with 128 internal units.
    model.add(GRU(output_dim=window_samples * 4, return_sequences=False, consume_less='mem'))
    model.add(Dropout(0.1))
    # model.add(Dense(window_samples * 4, input_dim=window_samples * 6, activation='relu'))
    model.add(Dense(window_samples * 2, activation='relu'))
    model.add(Dropout(0.2))
    model.add(Dense(window_samples, activation='relu'))
    model.add(Dense(11, activation='softmax'))

    # Compile model
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

    if params['digital_only']:
        filepath = "weights.best_" + ori_accel_file.replace("_accel", "").replace(".txt", "_X") + "_digitonly_mlp.hdf5"
    else:
        filepath = "weights.best_" + ori_accel_file.replace("_accel", "").replace(".txt", "_X") + "_mlp.hdf5"
    # Fit the model
    verbosity = 0
    if verbose:
        verbosity = 1
    if params['test_experiment'] == False:
        # checkpoint
        #filepath = "weights-improvement-{epoch:02d}-{val_accuracy:.2f}.hdf5"
        checkpoint = ModelCheckpoint(filepath, monitor='val_accuracy', verbose=1, save_best_only=True, mode='max')
        callbacks_list = [checkpoint]

        model.fit(X_train, Y_train, validation_data=(X_test, Y_test), nb_epoch=params['nb_epoch'], batch_size=params['batch_size'], callbacks=callbacks_list, verbose=verbosity)

    model.load_weights(filepath)
    # print predictions
    if print_predictions and verbose:
        for x_val, y_val in zip(X_test, Y_test):
            print (y_val, ":", model.predict(np.array([x_val]), verbose=0))

    # Show histogram of data
    if graph_error_dist:
        pred = model.predict_classes(X_test)
        print("total numbers:", len(Y_test_eval), len(pred))
        if params['test_digital_only']:
            new_pred = []
            new_test = []
            for i in range(len(Y_test_eval)):
                if Y_test_eval[i] == 10:
                    continue
                else:
                    new_pred.append(pred[i])
                    new_test.append(Y_test_eval[i])
            pred = new_pred
            Y_test_eval = new_test
        print("total numbers:", len(Y_test_eval), len(pred))
        print("pred:", pred)
        print("Y_test", Y_test_eval)
        print("accuracy:", accuracy_score(Y_test_eval, pred))

#        plt.hist(K.eval(in_distance(K.constant(Y_test_eval), K.constant(pred))), bins=20, normed=True)
        # plt.show()

    return dict(zip(model.metrics_names, model.evaluate(X_test, Y_test, verbose=verbosity)))


def learn_location_mlp_softmax(accel_file, gyro_file, params, verbose=True):
    seed = 12
    np.random.seed(seed)

    ori_accel_file = accel_file

    if params['test_experiment'] == True:
        accel_file = "/home/sunlichao/Documents/lehigh_cloud/CCS21/data_test/sample_" + params["test_file_id"] + "/accel.txt"
        gyro_file = "/home/sunlichao/Documents/lehigh_cloud/CCS21/data_test/sample_" + params["test_file_id"] + "/gyro.txt"

    if params['first_data_load'] == True:
        # read the data in
        data = read_data(accel_file, gyro_file)

        # find windows where touching
        touching_windows, touching_labels = get_touching_windows(data, with_labels=True)
        expanded_touching_windows = expand_windows_interpolated(data, touching_windows)

        # convert to feature vectors
        positive_feature_vectors = feature_vectors_from_windows(expanded_touching_windows)

        # fix random seed for reproducibility

        # split into input (X) and output (Y) variables
        X = np.array(map(np.array, positive_feature_vectors))
        Y = np.array(map(np.array, touching_labels))
        np.save(accel_file.replace("_accel", "").replace(".txt", "_X"), X)
        np.save(accel_file.replace("_accel", "").replace(".txt", "_Y"), Y)

    else:  # 1handtyping1hour_accel
        X = np.load(accel_file.replace("_accel", "").replace(".txt", "_X.npy"))
        Y = np.load(accel_file.replace("_accel", "").replace(".txt", "_Y.npy"))

    X, Y = data_transform(X, Y, True, params['digital_only'])
    print("shapes: ", X.shape, Y.shape)

    print("shapes: ", X[0])

    if params['test_experiment'] == True:
        X_train, X_test, Y_train, Y_test = [], X, [], Y
    else:
        X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=params['test_size'])

    print("X test shapes: ", X_test.shape)

    if params['sampled'] > 0:
        X_train, Y_train = X_train[:int(len(X_train)*params['sampled']/60)], Y_train[:int(len(X_train)*params['sampled']/60)]
        print("XXX minutes X Y shapes: ", params['sampled'], X_train.shape, Y_train.shape)

    Y_test_eval = np.array([np.where(r==1)[0][0] for r in Y_test])

    # create model
    model = Sequential()
    model.add(Dense(window_samples * 4, input_dim=window_samples * 6, activation='relu'))
    model.add(Dropout(0.2))
    model.add(Dense(window_samples * 2, activation='relu'))
    model.add(Dense(window_samples, activation='relu'))
    model.add(Dense(11, activation='softmax'))

    # Compile model
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

    if params['digital_only']:
        filepath = "weights.best_" + ori_accel_file.replace("_accel", "").replace(".txt", "_X") + "_digitonly_mlp.hdf5"
    else:
        filepath = "weights.best_" + ori_accel_file.replace("_accel", "").replace(".txt", "_X") + "_mlp.hdf5"
    # Fit the model
    verbosity = 0
    if verbose:
        verbosity = 1
    if params['test_experiment'] == False:
        # checkpoint
        #filepath = "weights-improvement-{epoch:02d}-{val_accuracy:.2f}.hdf5"
        checkpoint = ModelCheckpoint(filepath, monitor='val_accuracy', verbose=1, save_best_only=True, mode='max')
        callbacks_list = [checkpoint]

        model.fit(X_train, Y_train, validation_data=(X_test, Y_test), nb_epoch=params['nb_epoch'], batch_size=params['batch_size'], callbacks=callbacks_list, verbose=verbosity)

    model.load_weights(filepath)
    # print predictions
    if print_predictions and verbose:
        for x_val, y_val in zip(X_test, Y_test):
            print (y_val, ":", model.predict(np.array([x_val]), verbose=0))

    # Show histogram of data
    if graph_error_dist:
        pred = model.predict_classes(X_test)
        print("total numbers:", len(Y_test_eval), len(pred))
        if params['test_digital_only']:
            new_pred = []
            new_test = []
            for i in range(len(Y_test_eval)):
                if Y_test_eval[i] == 10:
                    continue
                else:
                    new_pred.append(pred[i])
                    new_test.append(Y_test_eval[i])
            pred = new_pred
            Y_test_eval = new_test
        print("total numbers:", len(Y_test_eval), len(pred))
        print("pred:", pred)
        print("Y_test", Y_test_eval)
        print("accuracy:", accuracy_score(Y_test_eval, pred))

#        plt.hist(K.eval(in_distance(K.constant(Y_test_eval), K.constant(pred))), bins=20, normed=True)
        # plt.show()

    return dict(zip(model.metrics_names, model.evaluate(X_test, Y_test, verbose=verbosity)))


def learn_location_gru_mse(accel_file, gyro_file, params, verbose=True):
    seed = 12
    np.random.seed(seed)

    ori_accel_file = accel_file

    if params['test_experiment'] == True:
        accel_file = "/home/sunlichao/Documents/lehigh_cloud/CCS21/data_test/sample_" + params["test_file_id"] + "/accel.txt"
        gyro_file = "/home/sunlichao/Documents/lehigh_cloud/CCS21/data_test/sample_" + params["test_file_id"] + "/gyro.txt"
        x = 0

    if params['first_data_load'] == True:
        # read the data in
        data = read_data(accel_file, gyro_file)

        # find windows where touching
        touching_windows, touching_labels = get_touching_windows(data, with_labels=True)
        expanded_touching_windows = expand_windows_interpolated(data, touching_windows)

        # convert to feature vectors
        positive_feature_vectors = feature_vectors_from_windows(expanded_touching_windows)

        # fix random seed for reproducibility

        # split into input (X) and output (Y) variables
        X = np.array(map(np.array, positive_feature_vectors))
        Y = np.array(map(np.array, touching_labels))
        np.save(accel_file.replace("_accel", "").replace(".txt", "_X"), X)
        np.save(accel_file.replace("_accel", "").replace(".txt", "_Y"), Y)

    else:  # 1handtyping1hour_accel
        X = np.load(accel_file.replace("_accel", "").replace(".txt", "_X.npy"))
        Y = np.load(accel_file.replace("_accel", "").replace(".txt", "_Y.npy"))

    X, Y = data_transform(X, Y, False, params['digital_only'])
    print("X Y shapes: ", X.shape, Y.shape)

    X = X.reshape(len(X), params['data_shape'], len(X[0]) / params['data_shape'])

    # X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.33)
    if params['test_experiment'] == True:
        # np.random.seed(int(params['test_file_id']))
        # idx = np.random.choice(np.arange(len(X)), 16, replace=False)
        # X_train, X_test, Y_train, Y_test = [], X[idx], [], Y[idx]
        X_train, X_test, Y_train, Y_test = [], X, [], Y
    else:
        X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=params['test_size'])

    # X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.33)

    if params['sampled'] > 0:
        X_train, Y_train = X_train[:int(len(X_train) * params['sampled'] / 60)], Y_train[:int(
            len(X_train) * params['sampled'] / 60)]
        print("XXX minutes X Y shapes: ", params['sampled'], X_train.shape, Y_train.shape)

    # create model
    model = Sequential()
    # Add a LSTM layer with 128 internal units.
    model.add(GRU(output_dim=window_samples * 4, return_sequences=False, consume_less='mem'))
    model.add(Dropout(0.1))
    # model.add(Dense(window_samples * 4, input_dim=window_samples * 6, activation='relu'))
    model.add(Dense(window_samples * 2, activation='relu'))
    model.add(Dropout(0.2))
    model.add(Dense(window_samples, activation='relu'))
    model.add(Dense(2, activation='linear'))

    # Compile model
    # model.compile(loss='mse', optimizer='adam', metrics=['mse', in_dist_mean])
    model.compile(loss='mse', optimizer='adam', metrics=['mse'])

    if params['digital_only']:
        filepath = "weights.best_" + ori_accel_file.replace("_accel", "").replace(".txt", "_X") + "_digitonly_gru_mse.hdf5"
    else:
        filepath = "weights.best_" + ori_accel_file.replace("_accel", "").replace(".txt", "_X") + "_gru_mse.hdf5"

    # Fit the model
    verbosity = 0
    if verbose:
        verbosity = 1
    if params['test_experiment'] == False:
        # checkpoint
        # filepath = "weights-improvement-{epoch:02d}-{val_accuracy:.2f}.hdf5"
        checkpoint = ModelCheckpoint(filepath, monitor='val_mse', verbose=1, save_best_only=True, mode='min')
        callbacks_list = [checkpoint]
        model.fit(X_train, Y_train, validation_data=(X_test, Y_test), nb_epoch=params['nb_epoch'],
                  batch_size=params['batch_size'], callbacks=callbacks_list, verbose=verbosity)

    # model.load_weights(filepath)
    model = load_model(filepath)
    # print predictions
    if print_predictions and verbose:
        for x_val, y_val in zip(X_test, Y_test):
            print (y_val, ":", model.predict(np.array([x_val]), verbose=0))

    # Show histogram of data
    if graph_error_dist:
        pred = model.predict(X_test)
        # plt.hist(K.eval(in_distance(K.constant(Y_test), K.constant(pred))), bins=20, normed=True)
        # plt.show()

    n = 0
    for x_val, y_val in zip(X_test, Y_test):
        print (y_val, ":", model.predict(np.array([x_val]), verbose=0))
        n += 1

    # graph predictions
    if graph_predictions and verbose:
        m = 1
        # m = 34
        # m = params['m']
        # n = params['n']
        # m = 22
        # n = 10
        # m = 1
        # n = 3187

        pred_data = zip(X_test, Y_test)

        accuracy = []
        distanceError = []
        original_Coordinates=[]
        predicted_Coordinates=[]
        for ii in xrange(min(m * n, len(X))):
            # for ii in xrange(len(X_test)):
            x_val, y_val = pred_data[ii]

            curr_plot = plt.subplot(m, n, ii + 1)  # the position parameter is 1-indexed

            plt.plot(y_val[0], -1 * y_val[1], 'go')
            pred = model.predict(np.array([x_val]), verbose=0).flatten()
            plt.plot(pred[0], -1 * pred[1], 'ro')

            if coordinatesToDigit(deNormX(y_val[0]), deNormY(y_val[1])) != "Not Numeric":
                # print("Coordinates: ",deNormX(y_val[0]) , deNormY(y_val[1]),"|",deNormX(pred[0]), deNormY(pred[1]))
                # print("Button: ",coordinatesToDigit(deNormX(y_val[0]) , deNormY(y_val[1])),"|",coordinatesToDigit(deNormX(pred[0]), deNormY(pred[1])))
                # print

                accuracy.append((coordinatesToDigit(deNormX(y_val[0]), deNormY(y_val[1])),
                                 coordinatesToDigit(deNormX(pred[0]), deNormY(pred[1]))))
            else:
                accuracy.append((coordinatesToDigit(deNormX(y_val[0]), deNormY(y_val[1])),
                                 coordinatesToDigit(deNormX(pred[0]), deNormY(pred[1]))))

            curr_plot.xaxis.set_visible(False)
            curr_plot.yaxis.set_visible(False)

            plt.ylim(-1, 1)
            plt.xlim(-1, 1)
            
            original_Coordinates.append((deNormX(y_val[0]), deNormY(y_val[1])) )
            predicted_Coordinates.append( (deNormX(pred[0]), deNormY(pred[1])) )
        # plt.show()

    print("original_Coordinates", original_Coordinates)
    print("predicted_Coordinates",predicted_Coordinates)
    if params['createANDValidateCCs']:
        createANDValidateCCs(original_Coordinates,predicted_Coordinates)

    correctPredictions = 0
    touch_isNumber = 0
    Y_test_now = []
    pred_test_now = []
    for item in accuracy:
        if params['test_digital_only']:
            if item[0] != "Not Numeric":
                touch_isNumber += 1
                Y_test_now.append(item[0])
                pred_test_now.append(item[1])
                if item[0] == item[1]:
                    correctPredictions += 1
        else:
            touch_isNumber += 1
            Y_test_now.append(item[0])
            pred_test_now.append(item[1])
            if item[0] == item[1]:
                correctPredictions += 1

    print(Y_test_now)
    print(pred_test_now)

    correctPredictionsPercentage = float(truediv(correctPredictions, touch_isNumber)) * 100
    print "Correct Predictions: " + str(correctPredictions) + "/" + str(touch_isNumber) + " | " + str(
        "{:.2f}".format(correctPredictionsPercentage)) + "%"
    print

    return correctPredictionsPercentage
    return dict(zip(model.metrics_names, model.evaluate(X_test, Y_test, verbose=verbosity)))


def learn_location(accel_file, gyro_file, params, verbose=True):
    seed = 12
    np.random.seed(seed)

    ori_accel_file = accel_file

    if params['test_experiment'] == True:
        accel_file = "/home/sunlichao/Documents/lehigh_cloud/CCS21/data_test/sample_" + params["test_file_id"] + "/accel.txt"
        gyro_file = "/home/sunlichao/Documents/lehigh_cloud/CCS21/data_test/sample_" + params["test_file_id"] + "/gyro.txt"

    if params['first_data_load'] == True:
        # read the data in
        data = read_data(accel_file, gyro_file)

        # find windows where touching
        touching_windows, touching_labels = get_touching_windows(data, with_labels=True)
        expanded_touching_windows = expand_windows_interpolated(data, touching_windows)

        # convert to feature vectors
        positive_feature_vectors = feature_vectors_from_windows(expanded_touching_windows)

        #fix random seed for reproducibility

        # split into input (X) and output (Y) variables
        X = np.array(map(np.array, positive_feature_vectors))
        Y = np.array(map(np.array, touching_labels))
        np.save(accel_file.replace("_accel", "").replace(".txt", "_X"), X)
        np.save(accel_file.replace("_accel", "").replace(".txt", "_Y"), Y)

    else: # 1handtyping1hour_accel
        X = np.load(accel_file.replace("_accel", "").replace(".txt", "_X.npy"))
        Y = np.load(accel_file.replace("_accel", "").replace(".txt", "_Y.npy"))

    X, Y = data_transform(X, Y, False, params['digital_only'])
    print("X Y shapes: ", X.shape, Y.shape)

    # X = X.reshape(len(X), params['data_shape'], len(X[0]) / params['data_shape'])

    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=params['test_size'])

    if params['sampled'] > 0:
        X_train, Y_train = X_train[:int(len(X_train) * params['sampled'] / 60)], Y_train[:int(
            len(X_train) * params['sampled'] / 60)]
        print("XXX minutes X Y shapes: ", params['sampled'], X_train.shape, Y_train.shape)

    # create model
    model = Sequential()
    model.add(Dense(window_samples * 4, input_dim=window_samples * 6, activation='linear'))
    model.add(Dense(window_samples * 2, activation='linear'))
    model.add(Dense(window_samples, activation='linear'))
    model.add(Dense(2, activation='linear'))

    # Compile model
    model.compile(loss='mse', optimizer='adam', metrics=['mse', in_dist_mean])

    if params['digital_only']:
        filepath = "weights.best_" + ori_accel_file.replace("_accel", "").replace(".txt", "_X") + "_digitonly_mlp.hdf5"
    else:
        filepath = "weights.best_" + ori_accel_file.replace("_accel", "").replace(".txt", "_X") + "_mlp.hdf5"

    # Fit the model
    verbosity = 0
    if verbose:
        verbosity = 1
    if params['test_experiment'] == False:
        # checkpoint
        # filepath = "weights-improvement-{epoch:02d}-{val_accuracy:.2f}.hdf5"
        checkpoint = ModelCheckpoint(filepath, monitor='val_mse', verbose=1, save_best_only=True, mode='min')
        callbacks_list = [checkpoint]
        model.fit(X_train, Y_train, validation_data=(X_test, Y_test), nb_epoch=params['nb_epoch'], batch_size=params['batch_size'], callbacks=callbacks_list,verbose=verbosity)

    model.load_weights(filepath)
    # print predictions
    if print_predictions and verbose:
        for x_val, y_val in zip(X_test, Y_test):
            print (y_val, ":", model.predict(np.array([x_val]), verbose=0))

    # Show histogram of data
    if graph_error_dist:
        pred = model.predict(X_test)
        #plt.hist(K.eval(in_distance(K.constant(Y_test), K.constant(pred))), bins=20, normed=True)
        #plt.show()

    n = 0
    for x_val, y_val in zip(X_test, Y_test):
        print (y_val, ":", model.predict(np.array([x_val]), verbose=0))
        n += 1

    # graph predictions
    if graph_predictions and verbose:
        m = 1
        #m = 34
        # m = params['m']
        # n = params['n']
        # m = 22
        # n = 10
        # m = 1
        # n = 3187

        pred_data = zip(X_test, Y_test)

        accuracy=[]
        distanceError=[]
        for ii in xrange(min(m * n, len(X))):
        # for ii in xrange(len(X_test)):
            x_val, y_val = pred_data[ii]

            curr_plot = plt.subplot(m, n, ii + 1) # the position parameter is 1-indexed

            plt.plot(y_val[0],  -1 * y_val[1], 'go')
            pred = model.predict(np.array([x_val]), verbose=0).flatten()
            plt.plot(pred[0], -1 *  pred[1], 'ro')

            if coordinatesToDigit(deNormX(y_val[0]), deNormY(y_val[1]))!="Not Numeric":
                # print("Coordinates: ",deNormX(y_val[0]) , deNormY(y_val[1]),"|",deNormX(pred[0]), deNormY(pred[1]))
                # print("Button: ",coordinatesToDigit(deNormX(y_val[0]) , deNormY(y_val[1])),"|",coordinatesToDigit(deNormX(pred[0]), deNormY(pred[1])))
                # print

                accuracy.append( (coordinatesToDigit(deNormX(y_val[0]) , deNormY(y_val[1])),coordinatesToDigit(deNormX(pred[0]), deNormY(pred[1]))) )
            else:
                accuracy.append((coordinatesToDigit(deNormX(y_val[0]), deNormY(y_val[1])),
                                 coordinatesToDigit(deNormX(pred[0]), deNormY(pred[1]))))

            curr_plot.xaxis.set_visible(False)
            curr_plot.yaxis.set_visible(False)

            plt.ylim(-1, 1)
            plt.xlim(-1, 1)

        #plt.show()

    correctPredictions=0
    touch_isNumber=0
    Y_test_now = []
    pred_test_now = []
    for item in accuracy:
        if params['test_digital_only']:
            if item[0]!="Not Numeric":
                touch_isNumber += 1
                Y_test_now.append(item[0])
                pred_test_now.append(item[1])
                if item[0] == item[1]:
                    correctPredictions += 1
        else:
            touch_isNumber += 1
            Y_test_now.append(item[0])
            pred_test_now.append(item[1])
            if item[0] == item[1]:
                correctPredictions += 1

    print(Y_test_now)
    print(pred_test_now)

    correctPredictionsPercentage = float(truediv(correctPredictions,touch_isNumber))*100
    print "Correct Predictions: "+str(correctPredictions)+"/"+str(touch_isNumber)+" | "+str("{:.2f}".format(correctPredictionsPercentage))+"%"
    print 
    
    return correctPredictionsPercentage
    return dict(zip(model.metrics_names, model.evaluate(X_test, Y_test, verbose=verbosity)))


def deNormX(normVal):
    denormVal=(normVal+1)*1080/2
    return denormVal


def deNormY(normVal):
    denormVal=(normVal+1)*2280/2
    return denormVal


def coordinatesToDigit(x,y):
    button= "Not Numeric"
    x = int(x)
    y = int(y)
    if x in range(0, 272) and y in range(1490,1650):
        button = "1"
    elif x in range(273, 538) and y in range(1490,1650):
        button = "2"
    elif x in range(545, 804) and y in range(1490,1650):
        button = "3"
    elif x in range(0, 272) and y in range(1655,1814):
        button = "4"
    elif x in range(273, 538) and y in range(1655,1814):
        button = "5"
    elif x in range(545, 804) and y in range(1655,1814):
        button = "6"
    elif x in range(0, 272) and y in range(1820,1981):
        button = "7"
    elif x in range(273, 538) and y in range(1820,1981):
        button = "8"
    elif x in range(545, 804) and y in range(1820,1981):
        button = "9"
    elif x in range(278, 538) and y in range(1985,2151):
        button = "0"
    return button


def closeDigits(x,y):
    def distance (p1,p2):
        distance = math.sqrt( ((p1[0]-p2[0])**2)+((p1[1]-p2[1])**2) )
        return distance
    
    buttonsList=[]
    
    point=(x,y)
    
    threshold=30
    
    #Calculate the coordinates of the center of each button
    buttonCenter={}
    buttonCenter[0] = ( ((278+538)/2) , ((1985+2151)/2))
    buttonCenter[1] = ( ((0+272)/2) , ((1490+1650)/2))
    buttonCenter[2] = ( ((273+538)/2) , ((1490+1650)/2)) 
    buttonCenter[3] = ( ((545+804)/2) , ((1490+1650)/2)) 
    buttonCenter[4] = ( ((0+272)/2) , ((1655+1814)/2)) 
    buttonCenter[5] = ( ((273+538)/2) , ((1655+1814)/2)) 
    buttonCenter[6] = ( ((545+804)/2) , ((1655+1814)/2)) 
    buttonCenter[7] = ( ((0+272)/2) , ((1820+1981)/2)) 
    buttonCenter[8] = ( ((273+538)/2) , ((1820+1981)/2)) 
    buttonCenter[9] = ( ((545+804)/2) , ((1820+1981)/2)) 

    #Calculate the distance between the predicted coordinates and the center of each button
    distanceFromButtons={}
    for key in buttonCenter:
        centerPoint=buttonCenter[key]
        distanceFromButtons[key]=distance(point,centerPoint)
        
    #Get the minimum distance    
    minDistance_Key = min(distanceFromButtons.keys(), key=(lambda k: distanceFromButtons[k]))
    minDistance=distance(point,buttonCenter[minDistance_Key])
    
    #Find buttons that are close to the predicted coordinates based on threshold
    for key in distanceFromButtons:
        if (abs(distanceFromButtons[key]-minDistance)< threshold ):
            buttonsList.append(str(key))
    return buttonsList

def get_cc_number():
    if len(sys.argv) < 2:
        usage()
        sys.exit(1)

    return sys.argv[1]

def sum_digits(digit):
    if digit < 10:
        return digit
    else:
        sum = (digit % 10) + (digit // 10)
        return sum

def validate(cc_num):
    # reverse the credit card number
    cc_num = cc_num[::-1]
    # convert to integer list
    cc_num = [int(x) for x in cc_num]
    # double every second digit
    doubled_second_digit_list = list()
    digits = list(enumerate(cc_num, start=1))
    for index, digit in digits:
        if index % 2 == 0:
            doubled_second_digit_list.append(digit * 2)
        else:
            doubled_second_digit_list.append(digit)

    # add the digits if any number is more than 9
    doubled_second_digit_list = [sum_digits(x) for x in doubled_second_digit_list]
    # sum all digits
    sum_of_digits = sum(doubled_second_digit_list)
    # return True or False
    return sum_of_digits % 10 == 0

def createANDValidateCCs(original_Coordinates,predicted_Coordinates):
    originalCreditCardNumber=""
    for item in original_Coordinates:
        originalCreditCardNumber+=coordinatesToDigit(item[0],item[1])
    
    print "--------------------------------"
    print "originalCreditCardNumber",originalCreditCardNumber, "Luhn_Valid",validate(originalCreditCardNumber)
    credidCards=[]
    for item in predicted_Coordinates:
        tmp=[]
        possible_digits = closeDigits(item[0],item[1])
        if len(credidCards)==0:
            for digit in possible_digits:
                tmp.append(str(digit))
        else:
            for card in credidCards:
                for digit in possible_digits:
                    tmp.append(str(card)+digit)
                    
        credidCards=list(tmp)
        
    print "total_number_of_items_in_list: ",len(credidCards)
    Luhn_Valid=[]
    for item in credidCards:
        if validate(item)==True:
            Luhn_Valid.append(item)
    print "total_number_of_items_pass_luhn:",len(Luhn_Valid)
    
    correct="no"
    for credit in Luhn_Valid:
        if credit==originalCreditCardNumber:
            correct="yes"
    print "is_one_of_passed_luhn_the_correct_cc_number:",correct
    print "--------------------------------"

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Backdoors')
    parser.add_argument('--digital_only', default=False)

    args = parser.parse_args()

    params = {}

    params['digital_only'] = False
    params['test_digital_only'] = False
    params['data_shape'] = 1
    params['output_dim'] = 8
    params['nb_epoch'] = 500
    params['batch_size'] = 30
    params['first_data_load'] = False
    params['test_experiment'] = True
    params['createANDValidateCCs'] = True
    params['sampled'] = 0
    params['test_file_id'] = "7"
    params['test_size'] = 0.33
    # params['m'] = 22
    # params['n'] = 10
    # params['m'] = 1
    # params['n'] = 3187

    for i in range(10):
        params['test_file_id'] = str(i)
        learn_location_gru_mse("100CCmerged_accel.txt", "100CCmerged_gyro.txt", params)
        # learn_location("100CCmerged_accel.txt", "100CCmerged_gyro.txt", params)


    #2Hands dataset, nb_epoch=500, batch_size=30
    # learn_location("diamantTest_accel_2hands.txt", "diamantTest_gyro_2hands.txt", params)
    # learn_location_gru_mse("diamantTest_accel_2hands.txt", "diamantTest_gyro_2hands.txt", params)
    # learn_location_mlp_softmax("diamantTest_accel_2hands.txt", "diamantTest_gyro_2hands.txt", params)
    # learn_location_gru_softmax("diamantTest_accel_2hands.txt", "diamantTest_gyro_2hands.txt", params)

    # learn_location("2handtyping1hour_accel.txt", "2handtyping1hour_gyro.txt", params)
    # learn_location_gru_mse("2handtyping1hour_accel.txt", "2handtyping1hour_gyro.txt", params)
    # learn_location_mlp_softmax("2handtyping1hour_accel.txt", "2handtyping1hour_gyro.txt", params)
    # learn_location_gru_softmax("2handtyping1hour_accel.txt", "2handtyping1hour_gyro.txt", params)


    #1Hand dataset, nb_epoch=50,batch_size=10
    # learn_location("diamantTest_accel_1hand.txt", "diamantTest_gyro_1hand.txt", params)
    # learn_location_gru_mse("diamantTest_accel_1hand.txt", "diamantTest_gyro_1hand.txt", params)
    # learn_location_mlp_softmax("diamantTest_accel_1hand.txt", "diamantTest_gyro_1hand.txt", params)
    # learn_location_gru_softmax("diamantTest_accel_1hand.txt", "diamantTest_gyro_1hand.txt", params)

    # learn_location("1handtyping1hour_accel.txt", "1handtyping1hour_gyro.txt", params)
    # learn_location_gru_mse("1handtyping1hour_accel.txt", "1handtyping1hour_gyro.txt", params)
    # learn_location_mlp_softmax("1handtyping1hour_accel.txt", "1handtyping1hour_gyro.txt", params)
    # learn_location_gru_softmax("1handtyping1hour_accel.txt", "1handtyping1hour_gyro.txt", params)