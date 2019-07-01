# https://datascience.stackexchange.com/questions/33364/why-model-fit-generator-in-keras-is-taking-so-much-time-even-before-picking-the
import tensorflow as tf
config = tf.ConfigProto()
config.gpu_options.allow_growth = True
session = tf.Session(config=config)


import numpy as np
import pandas as pd
from glob import glob
from pathlib import Path

from skimage.io import imread
import tensorflow.keras as keras
from skimage.transform import resize
from tensorflow.keras.optimizers import SGD
from tensorflow.python.keras.utils.data_utils import Sequence
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.layers import (Conv2D, MaxPooling2D, Dense, ReLU, Dropout, 
                                     Flatten, GlobalAveragePooling2D)

import warnings
warnings.filterwarnings("ignore")


def build_model(kernel_config, layer_config):
    inputs = keras.Input(shape=(3, 224, 224))
    x = inputs
    
    # add vgg blocks
    for i in range(len(layer_config)):
        for j in range(layer_config[i]):
            kernel_size = kernel_config[i]
            x = Conv2D(kernel_size, (3, 3), (1, 1), activation='relu', data_format='channels_first',
                       padding='same', name=f'block{i+1}_conv{j+1}')(x)
        x = MaxPooling2D((2, 2), strides=(2, 2), data_format='channels_first',
                         padding='valid', name=f'block{i+1}_pool')(x)
    
    # add dense layers
    x = Flatten(data_format='channels_first', name='flatten')(x)
    x = Dense(4096, activation='relu')(x)
    x = Dropout(0.5)(x)
    x = Dense(4096, activation='relu')(x)
    x = Dropout(0.5)(x)
    predictions = Dense(2622)(x)
    
    # define model
    model = keras.Model(inputs=inputs, outputs=predictions)
    
    return model


def edit_model(model):
    # Store the fully connected layers
    fc = model.get_layer('dense_1')

    # change the output layer
    fc_out = Dense(1, activation='sigmoid', name='output')

    # Reconnect the layers
    x = fc.output
    predictions = fc_out(x)

    # Create a new model
    gender_model = keras.Model(inputs=model.input, outputs=predictions)
        
    return gender_model


def make_img_df(paths_F, paths_M):
    df_F = pd.DataFrame(paths_F, columns=['fpath'])
    df_F['gender'] = 'f'

    df_M = pd.DataFrame(paths_M, columns=['fpath'])
    df_M['gender'] = 'm'

    df = pd.concat([df_F, df_M], axis=0)
    df = df.sample(frac=1).reset_index(drop=True)

    return df


def prepare_data(img_path):
    train_paths_F = glob(str(img_path/'aligned/*F/*.jpg'), recursive=True)
    train_paths_M = glob(str(img_path/'aligned/*M/*.jpg'), recursive=True)
    val_paths_F = glob(str(img_path/'valid/*F/*.jpg'), recursive=True)
    val_paths_M = glob(str(img_path/'valid/*M/*.jpg'), recursive=True)

    # make train df
    train_df = make_img_df(train_paths_F, train_paths_M)
    val_df = make_img_df(val_paths_F, val_paths_M)

    print('Data preparation completed!')

    return train_df, val_df


if __name__ == "__main__":

    kernel_config = [64, 128, 256, 512, 512]
    layer_config = [2, 2, 3, 3, 3]
    img_path = Path('./data/combined/')
    batch_size = 64

    # build model in tf based on the PyTorch architecture
    model = build_model(kernel_config, layer_config)
    model.load_weights('keras_weights.h5')

    # add Dense layers for binary classification task
    gender_model = edit_model(model)

    # freeze the pre-trained weights
    for layer in gender_model.layers[:-1]:
        layer.trainable = False

    # make train/val dataframe
    train_df, val_df = prepare_data(img_path)
    print(f'Train Dataframe shape: {train_df.shape}\nValid Dataframe shape: {val_df.shape}')

    # Data Generator
    train_datagen = ImageDataGenerator(
            featurewise_center=True,
            featurewise_std_normalization=True,
            rotation_range=10,
#             width_shift_range=0.2,
#             height_shift_range=0.2,
#             shear_range=0.2,
#             zoom_range=0.3,
            horizontal_flip=True,
            data_format='channels_first')
    val_datagen = ImageDataGenerator(data_format='channels_first')

    train_generator = train_datagen.flow_from_dataframe(
            train_df,
            x_col='fpath',
            y_col='gender',
            target_size=(224, 224),
            batch_size=batch_size,
            class_mode='binary')
    val_generator = val_datagen.flow_from_dataframe(
            val_df,
            x_col='fpath',
            y_col='gender',
            target_size=(224, 224),
            batch_size=batch_size,
            class_mode='binary')

    # Training
    num_epochs = 10
#     learning_rate = 0.001
#     decay_rate = learning_rate / num_epochs
#     momentum = 0.85
#     sgd = SGD(lr=learning_rate, momentum=momentum, decay=decay_rate, nesterov=False)

    gender_model.compile(loss='binary_crossentropy', 
                         optimizer='adam', 
                         metrics=['accuracy'])
    print(gender_model.summary())
    
    
    print('Start training...')
    gender_model.fit_generator(generator=train_generator,
#                                steps_per_epoch=500,
                               epochs=num_epochs,
                               verbose=1,
                               validation_data=val_generator,
                               use_multiprocessing=True,
                               workers=4,
                               max_queue_size=16)
    print('Model training completed!')

    # save model
    gender_model.save('gender_cls_model.h5')
