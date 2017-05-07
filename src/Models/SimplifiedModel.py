from keras.models import Sequential
from keras.layers.core import Flatten, Dense, Dropout
from keras.layers.convolutional import Convolution2D
from keras.layers import Lambda, Cropping2D
from keras.layers.normalization import BatchNormalization
from keras.regularizers import l2


def SimplifiedModel(FLAGS, input_img_shape):

    assert FLAGS.top_crop is not None, "FLAGS.top_crop wasn't provided"
    assert FLAGS.bottom_crop is not None, "FLAGS.bottom_crop wasn't provided"
    assert FLAGS.width is not None, "FLAGS.width wasn't provided"


    model = Sequential()
    model.add(Cropping2D(cropping=((FLAGS.top_crop,FLAGS.bottom_crop),(0,0)), input_shape=input_img_shape))
    model.add(Lambda(lambda x: (x/127.5)-1.0))
    model.add(Convolution2D(int(16 * FLAGS.width), 3, 3, border_mode='same', activation='relu', W_regularizer=l2(FLAGS.weight_decay)))
    model.add(Convolution2D(int(16 * FLAGS.width), 3, 3, subsample=(2,2), activation='relu', W_regularizer=l2(FLAGS.weight_decay)))
    #model.add(Dropout(0.25))
    model.add(BatchNormalization())
    model.add(Convolution2D(int(32 * FLAGS.width), 3, 3, border_mode='same', activation='relu', W_regularizer=l2(FLAGS.weight_decay)))
    model.add(Convolution2D(int(32 * FLAGS.width), 3, 3, subsample=(2,2), activation='relu', W_regularizer=l2(FLAGS.weight_decay)))
    #model.add(Dropout(0.25))
    model.add(BatchNormalization())
    model.add(Convolution2D(int(64 * FLAGS.width), 3, 3, border_mode='same', activation='relu', W_regularizer=l2(FLAGS.weight_decay)))
    model.add(Convolution2D(int(64 * FLAGS.width), 3, 3, subsample=(2,2), activation='relu', W_regularizer=l2(FLAGS.weight_decay)))
    #model.add(Dropout(0.25))
    model.add(BatchNormalization())
    model.add(Convolution2D(int(64 * FLAGS.width), 3, 3, border_mode='same', activation='relu', W_regularizer=l2(FLAGS.weight_decay)))
    model.add(Convolution2D(int(64 * FLAGS.width), 3, 3, subsample=(2,2), activation='relu', W_regularizer=l2(FLAGS.weight_decay)))
    model.add(BatchNormalization())
    #model.add(Dropout(0.25))
    model.add(Convolution2D(int(128 * FLAGS.width), 3, 3, border_mode='same', activation='relu', W_regularizer=l2(FLAGS.weight_decay)))
    model.add(Convolution2D(int(128 * FLAGS.width), 3, 3, subsample=(2,2), activation='relu', W_regularizer=l2(FLAGS.weight_decay)))
    model.add(BatchNormalization())
    model.add(Flatten())
    model.add(Dropout(0.25))
    model.add(Dense(128, activation='relu', W_regularizer=l2(FLAGS.weight_decay)))
    model.add(Dense(64, activation='relu', W_regularizer=l2(FLAGS.weight_decay)))
    model.add(Dense(16, activation='relu', W_regularizer=l2(FLAGS.weight_decay)))
    model.add(Dense(1, activation='linear', W_regularizer=l2(FLAGS.weight_decay)))


    return model
