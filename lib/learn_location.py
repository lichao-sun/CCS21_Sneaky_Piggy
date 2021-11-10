import sys
from operator import truediv
from axolotl import *
import numpy as np
import matplotlib.pyplot as plt
from keras.models import Sequential
from keras.layers import Dense, GRU, Embedding, Dropout, LSTM
from sklearn.model_selection import train_test_split
from keras import backend as K
from sklearn.metrics import accuracy_score
from keras.callbacks import ModelCheckpoint
import argparse

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

    if params['first'] == True:
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

    X = X.reshape(len(X), params['data_shape'], len(X[0])/params['data_shape'])
    # X = np.transpose(X, (0, 2, 1))
    print(X[0])
    print("shapes: ", X.shape)

    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.33)

    Y_test_eval = np.array([np.where(r==1)[0][0] for r in Y_test])


    # create model
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
    # model.compile(loss='mse', optimizer='adam', metrics=['mse', in_dist_mean])
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

    # checkpoint
    #filepath = "weights-improvement-{epoch:02d}-{val_accuracy:.2f}.hdf5"
    filepath = "weights.best.hdf5"
    checkpoint = ModelCheckpoint(filepath, monitor='val_accuracy', verbose=1, save_best_only=True, mode='max')
    callbacks_list = [checkpoint]

    # Fit the model
    verbosity = 0
    if verbose:
        verbosity = 1
    model.fit(X_train, Y_train, validation_data=(X_test, Y_test), nb_epoch=params['nb_epoch'], batch_size=params['batch_size'], callbacks=callbacks_list, verbose=verbosity)

    model.load_weights("weights.best.hdf5")

    # print predictions
    if print_predictions and verbose:
        for x_val, y_val in zip(X_test, Y_test):
            print (y_val, ":", model.predict(np.array([x_val]), verbose=0))

    # Show histogram of data
#     if graph_error_dist:
#         pred = model.predict_classes(X_test)
#         print("pred:", pred)
#         print("Y_test",Y_test_eval)
#         print("accuracy:", accuracy_score(Y_test_eval, pred))
# #        plt.hist(K.eval(in_distance(K.constant(Y_test_eval), K.constant(pred))), bins=20, normed=True)
#         # plt.show()

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

    return dict(zip(model.metrics_names, model.evaluate(X_test, Y_test, verbose=verbosity)))


def learn_location_mlp_softmax(accel_file, gyro_file, params, verbose=True):
    seed = 12
    np.random.seed(seed)

    if params['first'] == True:
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
    # print("shapes: ", X.shape, Y.shape)
    #
    # print("shapes: ", X[0])

    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.33)

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

    # checkpoint
    #filepath = "weights-improvement-{epoch:02d}-{val_accuracy:.2f}.hdf5"
    filepath = "weights.best.hdf5"
    checkpoint = ModelCheckpoint(filepath, monitor='val_accuracy', verbose=1, save_best_only=True, mode='max')
    callbacks_list = [checkpoint]

    # Fit the model
    verbosity = 0
    if verbose:
        verbosity = 1
    model.fit(X_train, Y_train, validation_data=(X_test, Y_test), nb_epoch=params['nb_epoch'], batch_size=params['batch_size'], callbacks=callbacks_list, verbose=verbosity)

    model.load_weights("weights.best.hdf5")
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


def learn_location(accel_file, gyro_file, params, verbose=True):
    seed = 12
    np.random.seed(seed)

    if params['first'] == True:
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
    model.fit(X_train, Y_train, validation_data=(X_test, Y_test), nb_epoch=params['nb_epoch'], batch_size=params['batch_size'], verbose=verbosity)

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


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Backdoors')
    parser.add_argument('--digital_only', default=False)

    args = parser.parse_args()

    params = {}

    params['digital_only'] = False
    params['test_digital_only'] = False
    params['data_shape'] = 1
    params['output_dim'] = 8
    params['first'] = False
    params['nb_epoch'] = 500
    params['batch_size'] = 30
    # params['m'] = 22
    # params['n'] = 10
    # params['m'] = 1
    # params['n'] = 3187


    #2Hands dataset, Best result: 41.84%, nb_epoch=500, batch_size=30
    # learn_location("diamantTest_accel_2hands.txt", "diamantTest_gyro_2hands.txt", params)
    # learn_location_mlp_softmax("diamantTest_accel_2hands.txt", "diamantTest_gyro_2hands.txt", params)
    # learn_location_gru_softmax("diamantTest_accel_2hands.txt", "diamantTest_gyro_2hands.txt", params)

    learn_location("2handtyping1hour_accel.txt", "2handtyping1hour_gyro.txt", params)
    # learn_location_mlp_softmax("2handtyping1hour_accel.txt", "2handtyping1hour_gyro.txt", params)
    # learn_location_gru_softmax("2handtyping1hour_accel.txt", "2handtyping1hour_gyro.txt", params)


    #1Hand dataset,Best result: 29.49%, nb_epoch=50,batch_size=10
    # learn_location("diamantTest_accel_1hand.txt", "diamantTest_gyro_1hand.txt", params)
    # learn_location_mlp_softmax("diamantTest_accel_1hand.txt", "diamantTest_gyro_1hand.txt", params)
    # learn_location_gru_softmax("diamantTest_accel_1hand.txt", "diamantTest_gyro_1hand.txt", params)

    # learn_location("1handtyping1hour_accel.txt", "1handtyping1hour_gyro.txt", params)
    # learn_location_mlp_softmax("1handtyping1hour_accel.txt", "1handtyping1hour_gyro.txt", params)
    # learn_location_gru_softmax("1handtyping1hour_accel.txt", "1handtyping1hour_gyro.txt", params)