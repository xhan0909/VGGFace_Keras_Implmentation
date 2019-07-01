from pathlib import Path
from glob import glob
import numpy as np
import pandas as pd

from skimage.io import imread
from skimage.transform import resize
import tensorflow.keras as keras
from tensorflow.keras.layers import (Conv2D, MaxPooling2D, Dense, ReLU, Dropout, 
                                     Flatten, GlobalAveragePooling2D)
from tensorflow.python.keras.utils.data_utils import Sequence
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.optimizers import SGD
from sklearn.metrics import classification_report


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


def make_img_df(paths_F, paths_M):
    df_F = pd.DataFrame(paths_F, columns=['fpath'])
    df_F['gender'] = 'f'

    df_M = pd.DataFrame(paths_M, columns=['fpath'])
    df_M['gender'] = 'm'

    df = pd.concat([df_F, df_M], axis=0)
    df = df.sample(frac=1).reset_index(drop=True)

    return df


if __name__ == "__main__":

    img_path = Path('./data/combined/')
    batch_size = 64
    
    # Recreate the exact same model, including weights and optimizer.
    gender_model = keras.models.load_model('gender_cls_model.h5')
    print(gender_model.summary())
    
    # make train/val dataframe
    train_df, val_df = prepare_data(img_path)
    
    # Valid generator
    val_datagen = ImageDataGenerator(data_format='channels_first')
    val_generator = val_datagen.flow_from_dataframe(val_df,
                                                    x_col='fpath',
                                                    y_col='gender',
                                                    target_size=(224, 224),
                                                    batch_size=batch_size,
                                                    class_mode='binary')
    
    # Overall loss and Accuracy
    print('Evaluating model...')
    overall_loss, overall_acc = gender_model.evaluate_generator(val_generator,
                                                                use_multiprocessing=True,
                                                                max_queue_size=10,
                                                                workers=4,
                                                                verbose=1)
    print(f'Overall Loss: {round(overall_loss, 3)}, Overall Accuracy: {round(overall_acc, 3)}\n')
    
    # Class accuracy
    print('Making predictions...')
    val_generator_all = val_datagen.flow_from_dataframe(
                            val_df,
                            x_col='fpath',
                            y_col='gender',
                            target_size=(224, 224),
                            batch_size=val_df.shape[0],
                            class_mode='binary')
    X, y = next(iter(val_generator_all))
    pred = gender_model.predict(X, verbose=1)
    pred = pred.squeeze()
    pred = [1 if p > 0.5 else 0 for p in pred]
    print(classification_report(y, pred, target_names=['Female', 'Male']))