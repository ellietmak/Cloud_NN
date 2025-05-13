import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications.resnet50 import ResNet50
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
import datetime
import os
import PIL
import argparse


class CloudNN():
    def __init__(self, path_to_images, labels_csv, input_shape_x, input_shape_y):
        self.path_to_images = path_to_images
        self.labels_csv = labels_csv
        self.input_shape_x = input_shape_x
        self.input_shape_y = input_shape_y
        self.im_gen = self.image_gen()
        self.model = self.model_structure()
    
    def image_gen(self):
        
        col_names = ['clear_sky', 'clear_sky_sun', 'low_clouds', 'low_clouds_sun', 
                     'low_clouds_total', 'low_clouds_total_sun', 'alto', 
                     'alto_sun', 'high_clouds', 'high_clouds_sun', 
                     'cannot_process']
        
        labels_df = pd.read_csv(self.labels_csv)

        train_80, test_20 = train_test_split(labels_df, test_size = 0.2)
        # DEFINE A TRAIN AND TEST SET
        train_df, test_df = train_test_split(train_80, test_size=0.2)

        # GENERATE THE IMAGES
        xtrain_intrain = ImageDataGenerator(horizontal_flip = True, vertical_flip = True, 
                                            rotation_range = 20).flow_from_dataframe(
            train_df,
            directory=self.path_to_images,
            x_col='filename',
            y_col=col_names,
        #     weight_col=None,
            target_size=(self.input_shape_x, self.input_shape_y),
            color_mode='rgb',
        #     classes=None,
            class_mode='raw',
            batch_size=30,
            shuffle=False,
            seed=None,
            save_to_dir=None,
            save_prefix='',
            save_format='jpg',
            subset=None,
            interpolation='nearest',
            validate_filenames=True,
        )

        xval_intrain = ImageDataGenerator(horizontal_flip = True, vertical_flip = True, 
                                        rotation_range = 20).flow_from_dataframe(
            test_df,
            directory=self.path_to_images,
            x_col='filename',
            y_col=col_names,
        #     weight_col=None,
            target_size=(self.input_shape_x, self.input_shape_y),
            color_mode='rgb',
        #     classes=None,
            class_mode='raw',
            batch_size=30,
            shuffle=False,
            seed=None,
            save_to_dir=None,
            save_prefix='',
            save_format='jpg',
            subset=None,
            interpolation='nearest',
            validate_filenames=True,
        )

        return xtrain_intrain, xval_intrain

    def model_structure(self):
        shape_x = self.input_shape_x
        shape_y = self.input_shape_y

        model = Sequential()
        pretrained_model = ResNet50(include_top = False, 
                                    input_shape = (shape_x, shape_y, 3), pooling = 'avg', 
                                    classes = 11, 
                                    weights='imagenet')

        model.add(pretrained_model)
        model.add(Flatten())
        model.add(Dense(5, activation='sigmoid')) 

        # Compile the model
        model.compile(optimizer='adam',
                        loss='mse',
                        metrics=[tf.keras.metrics.RootMeanSquaredError(), 'accuracy'])

        print(model.summary())

        return model
    
    def model_fit(self, epochs):
        model = self.model
        
        run_time_string = datetime.datetime.utcnow().isoformat(timespec='minutes')

        # define path to save model
        model_path = f'/nn_results/cloud_nn_{run_time_string}.h5'
        print(f"Training ... {model_path}")
        model.save(model_path)

        logdir = os.path.join("/nn_results", datetime.datetime.now().strftime("%Y%m%d-%H%M%S"))
        tensorboard_callback = tf.keras.callbacks.TensorBoard(logdir, histogram_freq=1)

        reduce_lr = tf.keras.callbacks.ReduceLROnPlateau(monitor='root_mean_squared_error', factor=0.75,
                                    patience=2, min_lr=1e-6, verbose=1, cooldown=0)

        csv_logger = tf.keras.callbacks.CSVLogger(f'/nn_results/training_{run_time_string}.log')

        earlystop = tf.keras.callbacks.EarlyStopping(monitor='root_mean_squared_error', min_delta=0.001,
                                                    patience=3,
                                                    verbose=1, mode='auto')

        model_check = tf.keras.callbacks.ModelCheckpoint(model_path,
                monitor='root_mean_squared_error',
                save_best_only=True,
                mode='min',
                verbose=1)

        xtrain_intrain = self.im_gen[0]
        xval_intrain = self.im_gen[1]

        model.fit(x=xtrain_intrain,
                epochs = epochs,
                validation_data=xval_intrain,
                callbacks=[tensorboard_callback, reduce_lr, csv_logger, earlystop, model_check])

        model.save('models/model_test.keras')
        
    
def main():
    parser = argparse.ArgumentParser(description='CNN to classify cloud images')
    parser.add_argument('path_to_images', type=str, help='path to the directory containing images')
    parser.add_argument('input_shape_x', type=tuple, help='image dimensions x')
    parser.add_argument('input_shape_y', type=tuple, help='image dimensions x')
    parser.add_argument('--labels_csv', type=str, help='path to the csv file containing image labels')
    parser.add_argument('--epochs', type=int, default=10, help='number of training epochs (default: 10)')
    parser.add_argument('-h', '--help', action='help', help='python3 nn_v5.py path_to_images input_shape_x input_shape_y --labels.csv --epochs')
    args = parser.parse_args()

    classifier = CloudNN(args.path_to_images, args.labels_csv, args.input_shape_x, args.input_shape_y)
    classifier.model_fit(args.epochs)

if __name__ == "__main__":
    main()

# READ FROM XML FILE MAYBE FOR ARUGMENTS