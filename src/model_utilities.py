import tensorflow as tf
from tensorflow.keras.models import Sequential, Model, load_model
from tensorflow.keras.layers import Conv2D, MaxPooling2D, AveragePooling2D, Activation, Dropout, Flatten, Dense, Dropout
import matplotlib.pyplot as plt
from tensorflow.keras.preprocessing import image
import datetime
import os
import numpy as np




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

    model.save("models/" + name)

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


def retrieve_data(datagen, TRAIN_BASE_DIRECTORY, BATCH_SIZE):

    train_generator = datagen.flow_from_directory(TRAIN_BASE_DIRECTORY,
                                                    target_size=(32, 32),
                                                    batch_size=BATCH_SIZE,
                                                    class_mode='categorical',
                                                    subset='training',
                                                    seed=5)

    val_generator = datagen.flow_from_directory(TRAIN_BASE_DIRECTORY,
                                                target_size=(32, 32),
                                                batch_size=BATCH_SIZE,
                                                class_mode='categorical',
                                                subset='validation',
                                                seed=5)

    return train_generator, val_generator

def prediction(choix, directory, model, plus_tilt, plus_hflip, plus_vhflip_bright):
    ###### Retrieving labels
    labels_array = os.listdir(directory)


    test_image = image.load_img(choix, target_size = (32, 32))
    plt.imshow(test_image)
    test_image = image.img_to_array(test_image)
    test_image = np.expand_dims(test_image, axis = 0)

    tf.autograph.set_verbosity(0)

    ###### Prediction with base model
    preds_base = model.predict_classes(test_image)
    prob_base = model.predict_proba(test_image)
    index_base = preds_base[0]

    ###### Predicting with Augmented model number 1
    preds_tilt = plus_tilt.predict_classes(test_image)
    prob_tilt = plus_tilt.predict_proba(test_image)
    index_tilt = preds_tilt[0]

    ###### Predicting with Augmented model number 2
    preds_hflip = plus_hflip.predict_classes(test_image)
    prob_hflip = plus_hflip.predict_proba(test_image)
    index_hflip = preds_hflip[0]

    ###### Predicting with Augmented model number 3
    preds_vhflip_bright = plus_vhflip_bright.predict_classes(test_image)
    prob_vhflip_bright = plus_vhflip_bright.predict_proba(test_image)
    index_vhflip_bright = preds_vhflip_bright[0]



    print(f"Ground truth : {choix[12:-9]}")
    print(f'Prediction of base model : {labels_array[index_base]}.', 'probs= ', prob_base[0][index_base] * 100) 
    print(f'Prediction of Augmented model #1 : {labels_array[index_tilt]}.', 'probs= ', prob_tilt[0][index_tilt] * 100)
    print(f'Prediction of Augmented model #2 : {labels_array[index_hflip]}.', 'probs= ', prob_hflip[0][index_hflip] * 100)
    print(f'Prediction of Augmented model #3 : {labels_array[index_vhflip_bright]}.', 'probs= ', prob_vhflip_bright[0][index_vhflip_bright] * 100)