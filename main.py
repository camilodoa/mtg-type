import json
import numpy as np
import requests
import shutil
import ssl
from time import sleep
import random
import os
from util import Config, Meta, Util
import tensorflow as tf
import keras
from keras.initializers import glorot_uniform
import keras_preprocessing
from keras.models import load_model
from keras_preprocessing import image
from keras.utils import CustomObjectScope
from keras_preprocessing.image import ImageDataGenerator
import matplotlib.pyplot as plt


def makeDirs():
    m = Meta()

    for data in ["test", "train"]:
        for type in m.types:
            directory = "{0}/{1}".format(data, type)
            if not os.path.exists(directory):
                os.makedirs(directory)

def download():
    with open('scryfall-default-cards.json') as json_file:
        cards = json.load(json_file)

        # ssl context
        ssl._create_default_https_context = ssl._create_unverified_context

        # Entry point used in case of crashes
        current = 0
        end = len(cards)

        util = Util()
        m = Meta()

        dict = {
                "Plains": 0,
                "Swamp" : 0,
                "Mountain" : 0,
                "Island" : 0,
                "Forest" : 0
        }

        for i in range(current, end):
            try:
                # Divide dataset into train and test data
                options = ["test"]*2 + ["train"]*8
                delegation = random.choice(options)

                # Ignore oversized and foil cards
                if cards[i]["oversized"] == True or cards[i]["foil"] == True:
                    continue

                # Ignore tokens
                if "Token" in cards[i]["type_line"]:
                    continue

                for type in m.types:
                    if type in cards[i]["type_line"]:
                        # Case for cards with repeated names, like lands
                        if dict.get(cards[i]["name"]) != None:
                            path = "{0}/{1}/{2}".format(delegation, type, util.cleanName(cards[i]["name"]+str(dict.get(cards[i]["name"]))))
                        else:
                            path = "{0}/{1}/{2}".format(delegation, type, util.cleanName(cards[i]["name"]))

                        r = requests.get(cards[i]["image_uris"]["small"], stream=True)
                        if r.status_code == 200:
                            with open(path, 'wb') as f:
                                r.raw.decode_content = True
                                shutil.copyfileobj(r.raw, f)

                        sleep(0.1)

                        if dict.get(cards[i]["name"]) != None:
                            print("Got {0} at index {1} for {2}ing. {3}% done".format(cards[i]['name']+str(dict.get(cards[i]["name"])), current, delegation, (current/end)*100))
                        else:
                            print("Got {0} at index {1} for {2}ing. {3}% done".format(cards[i]['name'], current, delegation, (current/end)*100))


                if dict.get(cards[i]["name"]) != None:
                    dict[cards[i]["name"]] = dict[cards[i]["name"]] + 1

                current += 1

            except:
                print('Error at ', current)
                print('Dict is: ', dict)
                print('Card is: ', cards[i]['name'])
                break

        print("Finished downloading")
        print('Current is ', current)
        print('Dict is: ', dict)

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
        # Flatten the results to feed into a DNN
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dropout(0.5),
        # 512 neuron hidden layer
        tf.keras.layers.Dense(512, activation='relu'),
        tf.keras.layers.Dense(7, activation='softmax')
    ])

    model.summary()

    model.compile(loss='categorical_crossentropy', optimizer='rmsprop', metrics=['accuracy'])

    history = model.fit_generator(train_generator, epochs=25, validation_data=validation_generator, verbose = 1)

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

def trial():
    with CustomObjectScope({'GlorotUniform': glorot_uniform()}):
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

def main():
    c = Config()

    if c.create_directories:
        makeDirs()

    if c.download:
        download()

    if c.train:
        train()

    if c.trial:
        trial()

if __name__ == '__main__':
    main()
