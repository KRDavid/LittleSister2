import tensorflow as tf
from tensorflow.keras.models import Sequential, Model, load_model
from tensorflow.keras.layers import Conv2D, MaxPooling2D, AveragePooling2D, Activation, Dropout, Flatten, Dense, Dropout
import datetime
import os

def train_model(model, train, val, name, n_epochs):
    model.compile(optimizer='adam',
                loss='categorical_crossentropy',
                metrics=['accuracy'])

    logdir = os.path.join("logs", datetime.datetime.now().strftime(name))
    tensorboard_callback = tf.keras.callbacks.TensorBoard(logdir, histogram_freq=1)

    model.fit_generator(train,
            epochs=n_epochs, 
            validation_data=(val), 
            callbacks=[tensorboard_callback],
            shuffle= False)

def create_model():
    model = Sequential()

    model.add(Conv2D(32, kernel_size=3, activation='relu', padding='same', input_shape=(32,32,3))) 
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dense(64, activation='relu'))
    # model.add(Dropout(0.2))

    model.add(Conv2D(64, kernel_size=3, padding='same', activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    # model.add(Dropout(0.2))


	#Fin obligatoire
    model.add(Flatten())
    model.add(Dense(10, activation='softmax'))

    model.summary()

    return model


def retrieve_data(datagen, TRAIN_BASE_DIRECTORY, VAL_BASE_DIRECTORY, BATCH_SIZE):

    train_generator = datagen.flow_from_directory(TRAIN_BASE_DIRECTORY,
                                                    target_size=(32, 32),
                                                    batch_size=BATCH_SIZE,
                                                    class_mode='categorical',
                                                    subset='training',
                                                    seed=5)

    val_generator = datagen.flow_from_directory(VAL_BASE_DIRECTORY,
                                                target_size=(32, 32),
                                                batch_size=BATCH_SIZE,
                                                class_mode='categorical',
                                                subset='training',
                                                seed=5)

    return train_generator, val_generator