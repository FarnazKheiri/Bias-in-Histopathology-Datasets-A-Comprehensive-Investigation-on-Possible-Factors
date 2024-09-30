import numpy as np
import tensorflow as tf
import os
import pathlib
import cv2
# import data_loader
import pandas as pd
import pdb
from split_image import split_image
from PIL import Image
import pickle
from preprocessing import get_labels
from dataBalancing import balancing
import random





# model dependencies
from tensorflow.keras.layers import Dense, Rescaling, MaxPooling2D, Flatten, Dropout
from tensorflow.keras.models import Model
from tensorflow.keras.models import Sequential
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.utils import to_categorical
from efficientnet.tfkeras import EfficientNetB0
from sklearn.neighbors import KNeighborsClassifier
from sklearn import metrics


# Define the model
base_model = EfficientNetB0(input_shape=(224, 224, 3), include_top=False, weights='imagenet', classes = 2)
num_classes = 2
batch_size = 16
model = Sequential()
model.add(Rescaling(scale=1./255, input_shape=(224, 224, 3)))
model.add(base_model)
for layer in base_model.layers[:-6]:
    layer.trainable = False

# Add a pooling layer, flatten the output and add a dense layer with 1024 units
model.add(MaxPooling2D((7,7)))
model.add(Flatten())
model.add(Dense(1024, activation="relu"))
# model.add(BatchNormalization())

# # Add regularization
model.add(Dropout(rate=0.3))
model.add(Dense(256, kernel_regularizer=tf.keras.regularizers.l2(0.01), activation="relu"))
# model.add(Dropout(rate=0.2))
# Add the output layer
model.add(Dense(2, activation="softmax"))

# # Compile the model
optimizer = Adam(learning_rate=0.001)
model.compile(optimizer=optimizer, loss='BinaryCrossentropy', metrics=['accuracy'])

# Train the model
history = model.fit(n_training_data,
                    n_training_cancer_labels,
                    steps_per_epoch=len(n_training_data) // batch_size,
                    epochs=5,
                    shuffle=True,
                    validation_data = (n_validation_data, n_validation_cancer_labels)
                   )