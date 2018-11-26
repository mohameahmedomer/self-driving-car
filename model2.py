import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from keras.models import Sequential
from keras.models import Model
import numpy as np
from keras.models import load_model
import os
import matplotlib.pyplot as plt

from utils2 import INPUT_SHAPE, batch_generator


from keras.models import load_model
from keras import metrics

import cv2
import matplotlib.image as mpimg
from keras.optimizers import Adam
from keras.callbacks import ModelCheckpoint
from keras.layers import Lambda, Conv2D, MaxPooling2D, Dropout, Dense, Flatten

import argparse
import os

np.random.seed(0)


def load_data(args):

    data_df = pd.read_csv(os.path.join(os.getcwd(), args.data_dir, 'driving_log.csv'), names=['center', 'left', 'right', 'steering','throttel','brake','speed'])

    X = data_df[['center', 'left', 'right']].values
    y = data_df[['steering','throttel','brake']].values

    X_train, X_valid, y_train, y_valid = train_test_split(X, y, test_size=args.test_size, random_state=0)

    return X_train, X_valid, y_train, y_valid


def build_model(args):

    model = Sequential()
    model.add(Lambda(lambda x: x/127.5-1.0, input_shape=(INPUT_SHAPE)))
    model.add(Conv2D(24, (5, 5),strides=(2, 2), activation='elu'))
    model.add(Conv2D(36, (5, 5),strides=(2, 2), activation='elu'))
    model.add(Conv2D(48, (5, 5), strides=(2, 2),activation='elu'))
    model.add(Conv2D(64, (3, 3), activation='elu'))
    model.add(Conv2D(64, (3, 3), activation='elu'))
    model.add(Conv2D(128, (3, 3), activation='elu'))
    model.add(Dropout(args.keep_prob))
    model.add(Flatten())
    model.add(Dense(100, activation='elu'))
    model.add(Dense(100, activation='elu'))
    model.add(Dense(50, activation='elu'))
    model.add(Dense(10, activation='elu'))
    model.add(Dense(3))

    model.summary()

    return model


def train_model(model, args, X_train, X_valid, y_train, y_valid):


    checkpoint = ModelCheckpoint('model-{epoch:03d}.h5', monitor=('val_loss'), verbose=0,save_best_only=args.save_best_only, mode='auto')

    def load_image(data_dir, image_file):

        return mpimg.imread(os.path.join(args.data_dir, image_file.strip()))
    i=0
    for index in np.random.permutation(X_train.shape[0]):
        center, left, right = X_train[index]
        image = load_image(args.data_dir, center)

        i += 1
        if i == args.batch_size:
            break



    model.compile(loss='mean_squared_error', optimizer=Adam(lr=args.learning_rate),metrics=['acc'])


    history = model.fit_generator(batch_generator(args.data_dir, X_train, y_train, 50, True),
                        steps_per_epoch=args.samples_per_epoch,
                        epochs=10,
                        max_queue_size=10,
                        validation_data=batch_generator(args.data_dir, X_valid, y_valid, 50, False),
                        validation_steps=len(X_valid)/10,
                        verbose=1)

    print(history.history.keys())
    #  "Accuracy"
    plt.plot(history.history['acc'])
    plt.plot(history.history['val_acc'])
    plt.title('model accuracy')
    plt.ylabel('accuracy')
    plt.xlabel('epoch')
    plt.legend(['train', 'validation'], loc='upper left')
    plt.show()
    # "Loss"
    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.title('model loss')
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.legend(['train', 'validation'], loc='upper left')
    plt.show()





def s2b(s):

    s = s.lower()
    return s == 'true' or s == 'yes' or s == 'y' or s == '1'

def main():
    
    parser = argparse.ArgumentParser(description='Behavioral Cloning Training Program')
    parser.add_argument('-d', help='data directory',        dest='data_dir',          type=str,   default='data')
    parser.add_argument('-t', help='test size fraction',    dest='test_size',         type=float, default=0.2)
    parser.add_argument('-k', help='drop out probability',  dest='keep_prob',         type=float, default=0.5)
    parser.add_argument('-n', help='number of epochs',      dest='nb_epoch',          type=int,   default=30)
    parser.add_argument('-s', help='samples per epoch',     dest='samples_per_epoch', type=int,   default= 30)
    parser.add_argument('-b', help='batch size',            dest='batch_size',        type=int,   default=10000)
    parser.add_argument('-o', help='save best models only', dest='save_best_only',    type=s2b,   default='true')
    parser.add_argument('-l', help='learning rate',         dest='learning_rate',     type=float, default=1.0e-4)
    args = parser.parse_args()



    data = load_data(args)

    model = build_model(args)
    train_model(model, args, *data)

if __name__ == '__main__':
    main()
