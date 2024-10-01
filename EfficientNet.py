
import tensorflow as tf
from tensorflow.keras.layers import Dense, Rescaling, MaxPooling2D, Flatten, Dropout
from tensorflow.keras.models import Sequential
from tensorflow.keras.optimizers import Adam
from efficientnet.tfkeras import EfficientNetB0


def EF_model(total_classes, training_data, training_cancer_labels, validation_data, validation_cancer_labels):
    # Define the model
    base_model = EfficientNetB0(input_shape=(224, 224, 3), include_top=False, weights='imagenet', classes = 2)
    num_classes = total_classes
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
    history = model.fit(training_data,
                        training_cancer_labels,
                        steps_per_epoch=len(training_data) // batch_size,
                        epochs=5,
                        shuffle=True,
                        validation_data = (validation_data, validation_cancer_labels)
                       )

    return model