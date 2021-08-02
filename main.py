import json
import numpy as np
import requests
import shutil
import ssl
from time import sleep
import random
import string
import os
import sys
from util import Config, Meta, Util

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.initializers import glorot_uniform
from tensorflow.keras.preprocessing import image
from tensorflow.keras.models import load_model
from tensorflow.keras.utils import CustomObjectScope
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import matplotlib.pyplot as plt

def download():
    with open('scryfall-default-cards.json') as json_file:
        cards = json.load(json_file)

        # ssl context
        ssl._create_default_https_context = ssl._create_unverified_context

        # Entry point used in case of crashes
        end = len(cards)

        util = Util()

        for i in range(0, end):
            try:
                # Ignore oversized and foil cards
                if cards[i]["oversized"] == True or cards[i]["foil"] == True:
                    print("Skipped {0} at index {1} because it's {2}".format(cards[i]['name'], i, "oversized" if cards[i] == True else "a foil"))  
                    continue

                if "image_uris" in cards[i]:
                    r = requests.get(cards[i]["image_uris"]["small"])
                    sleep(0.1)

                    if r.status_code != 200:
                        print("Status code is {0} for {1} at index {2}".format(r.status_code, cards[i]['name'], i))
                        continue

                    # Extract types from typeline
                    types = list(filter(lambda x : x[0].isalnum(), cards[i]["type_line"].split(" ")))
                    for card_type in types:
                        # Select where this image is going to go with 80% prob of training set
                        options = ["test"]*2 + ["train"]*8
                        delegation = random.choice(options)

                        if not os.path.exists(delegation):
                            os.makedirs(delegation)

                        directory = "{0}/{1}".format(delegation, card_type)

                        if not os.path.exists(directory):
                            os.makedirs(directory)

                        path = "{0}/{1}/{2}".format(delegation, card_type, "{1}{0}".format(util.cleanName(cards[i]["name"]), str(i)))
                        
                        # Save
                        with open(path, 'wb') as f:
                            f.write(r.content)
                            f.close()

                        print("Saved {0} {1} at index {2} for {3}ing. {4}% done".format(card_type, cards[i]['name'], i, delegation, (i/end)*100))
                
                else:
                    # If there are no root level image uris, it's a double faced card
                    for face in cards[i]["card_faces"]:
                        # Download and save each face
                        r = requests.get(face["image_uris"]["small"])
                        sleep(0.1)
                        if r.status_code != 200:
                            print("Status code is {0} for {1} at index {2}".format(r.status_code, face['name'], i))
                            continue

                        # Extract types from typeline
                        types = list(filter(lambda x : x[0].isalnum(), face["type_line"].split(" ")))
                        for card_type in types:
                            # Select where this image is going to go with 80% prob of training set
                            options = ["test"]*2 + ["train"]*8
                            delegation = random.choice(options)

                            if not os.path.exists(delegation):
                                os.makedirs(delegation)

                            directory = "{0}/{1}".format(delegation, card_type)

                            if not os.path.exists(directory):
                                os.makedirs(directory)

                            path = "{0}/{1}/{2}".format(delegation, card_type, "{1}{0}".format(util.cleanName(cards[i]["name"]), str(i)))
                            
                            # Save
                            with open(path, 'wb') as f:
                                f.write(r.content)
                                f.close()

                            print("Saved {0} {1} at index {2} for {3}ing. {4}% done".format(card_type, cards[i]['name'], i, delegation, (i/end)*100))

            except:
                print("Error: ", sys.exc_info()[0])
                print('Error at: ', i)
                print('Card is: ', cards[i]['name'])
                raise

        print("Finished downloading")
        print('Index is: ', i)

def train():
    TRAINING_DIR = "./train/"
    training_datagen = ImageDataGenerator(
      rescale = 1./255,
	  rotation_range=40,
      width_shift_range=0.2,
      height_shift_range=0.2,
      shear_range=0.2,
      zoom_range=0.2,
      horizontal_flip=True,
      fill_mode='nearest')

    VALIDATION_DIR = "./test/"
    validation_datagen = ImageDataGenerator(rescale = 1./255)

    train_generator = training_datagen.flow_from_directory(
    	TRAINING_DIR,
    	target_size=(146, 204),
    	class_mode='categorical'
    )

    validation_generator = validation_datagen.flow_from_directory(
    	VALIDATION_DIR,
    	target_size=(146, 204),
    	class_mode='categorical'
    )

    model = tf.keras.models.Sequential([
        # Note the input shape is the desired size of the image 146x204 with 3 bytes color
        # This is the first convolution
        tf.keras.layers.Conv2D(64, (3,3), activation='relu', input_shape=(146, 204, 3)),
        tf.keras.layers.MaxPooling2D(2, 2),

        # The second convolution
        tf.keras.layers.Conv2D(64, (3,3), activation='relu'),
        tf.keras.layers.MaxPooling2D(2,2),

        # The third convolution
        tf.keras.layers.Conv2D(128, (3,3), activation='relu'),
        tf.keras.layers.MaxPooling2D(2,2),

        # The fourth convolution
        tf.keras.layers.Conv2D(128, (3,3), activation='relu'),
        tf.keras.layers.MaxPooling2D(2,2),

        # Flatten the results to feed into a dropout layer
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dropout(0.5),

        # 512 neuron linear layer
        tf.keras.layers.Dense(512, activation='relu'),
        tf.keras.layers.Dense(7, activation='softmax')
    ])

    model.summary()

    model.compile(loss='categorical_crossentropy', optimizer='rmsprop', metrics=['accuracy'])

    history = model.fit(train_generator, epochs=25, validation_data=validation_generator, verbose = 1)

    model.save("mtg-types.h5")

    acc = history.history['acc']
    val_acc = history.history['val_acc']
    loss = history.history['loss']
    val_loss = history.history['val_loss']

    epochs = range(len(acc))

    plt.plot(epochs, acc, 'r', label='Training accuracy')
    plt.plot(epochs, val_acc, 'b', label='Validation accuracy')
    plt.title('Training and validation accuracy')
    plt.legend(loc=0)
    plt.figure()
    plt.show()

    plt.plot(epochs, loss, 'b', label='Training loss')
    plt.plot(epochs, val_loss, 'b', label='Validation loss')
    plt.title('Training and validation loss')
    plt.legend(loc=0)
    plt.figure()
    plt.show()

def trial():
    # load in model and show sample guesses
    model = load_model('mtg-types.h5')
    path = './trials/{0}'
    m = Meta()
    class_names = m.types # Reads all the folders in which images are present
    class_names = sorted(class_names) # Sorting them
    name_id_map = dict(zip(range(len(class_names)),class_names))

    for test_image in m.test_images:
        img = image.load_img(path.format(test_image), target_size=(146, 204))
        x = image.img_to_array(img)
        x = np.expand_dims(x, axis=0)

        images = np.vstack([x])
        classes = model.predict_classes(images, batch_size=10)
        print(test_image)
        print(classes, " ", name_id_map[classes[0]])
        img.show()
        sleep(4)
  

if __name__ == '__main__':
    c = Config()

    if c.download:
        download()

    if c.train:
        train()

    if c.trial:
        trial()
