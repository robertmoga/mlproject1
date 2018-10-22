import pandas as pd
import numpy as np
import time
import os
import keras
from keras.models import Sequential, load_model
from keras.layers import Input, Dense, Dropout, Flatten
from keras.layers import Conv2D, MaxPool2D, BatchNormalization
from keras import backend as K
import tensorboard
import cv2

DATASET_PATH = "resources/letters_dataset.csv"
MODEL_NAME = "out/keras_model.h5"
EPOCHS = 2
BATCH_SIZE = 128


def plotData(image, winname='window'):
    cv2.namedWindow(winname)
    cv2.moveWindow(winname,50,50)
    cv2.imshow(winname, image)

    cv2.waitKey(0)
    cv2.destroyAllWindows()


def load_dataset():
    dataset = pd.read_csv(DATASET_PATH).as_matrix()
    dataset = np.array(dataset)
    np.random.seed(7)
    np.random.shuffle(dataset)

    data_split_factor = int(len(dataset) * 0.8)
    print(str(len(dataset)) + "   " + str(dataset.shape) + "  " + str(dataset.size))

    x_train = dataset[:data_split_factor, 1:]
    y_train = dataset[:data_split_factor, 0]

    x_test = dataset[data_split_factor:, 1:]
    y_test = dataset[data_split_factor:, 0]

    del dataset

    # reshaping
    x_train = x_train.reshape(x_train.shape[0], 64, 64, 1)
    x_test = x_test.reshape(x_test.shape[0], 64, 64, 1)
    print(">>>" + str(x_train.shape[0]) + " \n " + str(x_train[0]))

    # turning every list into a nparray
    x_train = np.array(x_train)
    y_train= np.array(y_train)
    x_test = np.array(x_test)
    y_test = np.array(y_test)

    # transposing labels by subtracting 1
    y_train -= 1
    y_test -= 1

    # one hot encoding
    y_train = keras.utils.to_categorical(y_train, 22)
    y_test = keras.utils.to_categorical(y_test, 22)

    print(">> Input Train shape : " + str(x_train.shape) + "  \n>> Validation Train shape : " + str(y_train.shape))

    return x_train, y_train, x_test, y_test

def build_model():
    model = Sequential()
    model.add(Conv2D(filters=64, kernel_size=3, strides=1,
                     padding='same', activation='relu',
                     input_shape=[64, 64, 1]))
    model.add(Conv2D(filters=64, kernel_size=3, strides=1,
                     padding='same', activation='relu',
                     input_shape=[64, 64, 1]))
    model.add(BatchNormalization())
    model.add(MaxPool2D(pool_size=2, strides=2, padding='same'))

    model.add(Conv2D(filters=128, kernel_size=3, strides=1,
                     padding='same', activation='relu'))
    model.add(Conv2D(filters=128, kernel_size=3, strides=1,
                     padding='same', activation='relu'))

    model.add(BatchNormalization())
    model.add(MaxPool2D(pool_size=2, strides=2, padding='same'))

    model.add(Flatten())
    model.add(Dense(1024, activation='relu'))
    model.add(Dense(22, activation='softmax'))

    return model


def save_model(model):
    model.save(MODEL_NAME)
    print(">> Model saved successfully")


def train(model, x_train, y_train, x_test, y_test):
    model.compile(loss=keras.losses.categorical_crossentropy,
                  optimizer=keras.optimizers.Adadelta(),
                  metrics=['accuracy'])

    earlyStopping = keras.callbacks.EarlyStopping(monitor='val_loss',
                            min_delta=0, patience=2, verbose=1, mode='auto')

    model.fit(x_train, y_train, batch_size=BATCH_SIZE,
              verbose=1, epochs=100, validation_data=(x_test, y_test),
              callbacks= [ keras.callbacks.TensorBoard(log_dir='./', histogram_freq=1,
                                                       write_graph=True, write_images=True), earlyStopping])
    save_model(model)


def run_prediction(x_test, y_test):
    labels = {1: 97, 2: 98, 3: 99, 4: 100, 5: 101, 6: 102, 7: 103,
              8: 104, 9: 105, 10: 106, 11: 108, 12: 109, 13: 110,
              14: 111, 15: 112, 16: 113, 17: 114, 18: 115, 19: 116,
              20: 117, 21: 118, 22: 122}

    model = load_model(MODEL_NAME)
    true_positives, true_negatives = 0,0

    for i in range(len(x_test)):
        x = x_test[i]
        y = y_test[i]
        x = x.reshape(1, 64, 64, 1)
        y1 = model.predict(x)
        predicted_label = np.argmax(y1, axis=1)
        print(">> STrange " + str(predicted_label[0]))
        y1 = int(predicted_label[0]) + 1

        if chr(y1) != chr(y):
            true_negatives += 1
            print(">>" + chr(y1) + "  " + str(y1) + " " + str(y))
            true_positives += 1

    print(" >> Pred bune : " + str((true_positives / len(x_test)) * 100))
    print(" >> Pred rele : " + str((true_negatives / len(x_test)) * 100))


def run_test():
    if not os.path.exists('out'):
        os.mkdir('out')

    timer = time.time()
    x_train, y_train, x_test, y_test = load_dataset()
    print(">> Time elapsed on loading data : " + str((time.time() - timer) / 60) + " minutes")

    model = build_model()
    timer = time.time()
    train(model, x_train, y_train, x_test, y_test)
    print(">> Time elapsed on training : " + str((time.time() - timer) / 60) + " minutes")

    new_model = load_model(MODEL_NAME)
    score = new_model.evaluate(x_test, y_test)
    try:
        with open("keras_train_results1.txt", 'a') as f:
            for i in range(len(new_model.metrics_names)):
                print(str(new_model.metrics_names[i]) + "   " + str(score[i]))
                f.write(str(new_model.metrics_names[i]) + "   " + str(score[i]))
            f.close()
    except Exception as e:
        print(">>Exception at writing the evaluate results" + str(e))


if __name__=="__main__":
    print(">> Strat")
    run_test()