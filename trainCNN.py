import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Activation, Dense, Flatten, BatchNormalization, Conv2D, MaxPool2D, Dropout
from tensorflow.keras.optimizers import Adam, SGD
from tensorflow.keras.metrics import categorical_crossentropy
from keras.preprocessing.image import ImageDataGenerator
import itertools
import random
import warnings
import numpy as np
import matplotlib.pyplot as plt
import cv2
from tensorflow.keras.callbacks import ReduceLROnPlateau
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping

warnings.simplefilter(action='ignore', category=FutureWarning)

# train_path = r'D:\CV Project\Sign Language Recognition\dataset\train'
# test_path = r'D:\CV Project\Sign Language Recognition\dataset\test'
# path = r'D:\Dataset\ASL\asl_alphabet_train\asl_alphabet_train'
# path = r'D:\CV Project\Sign Language Recognition\dataset\all'
# path = r'D:\Dataset\ASL Seg\asl_dataset\asl_dataset'
path = r'D:\Dataset\ASL Gray\all'

train_datagen = ImageDataGenerator(
    rotation_range=15,
    rescale=1 / 255,
    zoom_range=0.1,
    horizontal_flip=True,
    width_shift_range=0.1,
    height_shift_range=0.1,
    validation_split=0.2)

train_batches = train_datagen.flow_from_directory(directory=path,
                                                  target_size=(128, 128),
                                                  class_mode='categorical',
                                                  color_mode ="grayscale",
                                                  batch_size=64,
                                                  subset='training')

test_batches = train_datagen.flow_from_directory(directory=path,
                                                 target_size=(128, 128),
                                                 class_mode='categorical',
                                                 color_mode ="grayscale",
                                                 batch_size=64,
                                                 subset='validation')

imgs, labels = next(train_batches)


# Plotting the images...
def plotImages(images_arr):
    fig, axes = plt.subplots(1, 10, figsize=(30, 20))
    axes = axes.flatten()
    for img, ax in zip(images_arr, axes):
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        ax.imshow(img)
        ax.axis('off')
    plt.tight_layout()
    plt.show()


# plotImages(imgs)
# print(imgs.shape)
# print(labels)

model = Sequential()

model.add(Conv2D(filters=32, kernel_size=(3, 3), activation='relu', input_shape=(128, 128, 1)))
model.add(MaxPool2D(pool_size=(2, 2), strides=2))

model.add(Conv2D(filters=64, kernel_size=(3, 3), activation='relu', padding="same"))
model.add(MaxPool2D(pool_size=(2, 2), strides=2))

model.add(Conv2D(filters=128, kernel_size=(3, 3), activation='relu', padding="valid"))
model.add(MaxPool2D(pool_size=(2, 2), strides=2))

model.add(Flatten())

# model.add(Dense(64, activation="relu"))
# model.add(Dense(128, activation="relu"))
# model.add(Dropout(0.2))
model.add(Dense(128, activation="relu"))
model.add(Dropout(0.2))
model.add(Dense(26, activation="softmax"))

model.compile(optimizer=Adam(learning_rate=0.001), loss='categorical_crossentropy', metrics=['accuracy'])
reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.2, patience=1, min_lr=0.0001)
early_stop = EarlyStopping(monitor='val_loss', min_delta=0, patience=2, verbose=0, mode='auto')

# model.compile(optimizer=SGD(learning_rate=0.001), loss='categorical_crossentropy', metrics=['accuracy'])
# reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.2, patience=1, min_lr=0.0005)
# early_stop = EarlyStopping(monitor='val_loss', min_delta=0, patience=2, verbose=0, mode='auto')

history2 = model.fit(train_batches, epochs=30, callbacks=[reduce_lr, early_stop], validation_data=test_batches)


imgs, labels = next(train_batches)  # For getting next batch of imgs...

imgs, labels = next(test_batches)  # For getting next batch of imgs...
scores = model.evaluate(imgs, labels, verbose=0)
print(f'{model.metrics_names[0]} of {scores[0]}; {model.metrics_names[1]} of {scores[1] * 100}%')

model.save('model2.h5')

print(history2.history)

imgs, labels = next(test_batches)

model = keras.models.load_model(r"model2.h5")

scores = model.evaluate(imgs, labels, verbose=0)
print(f'{model.metrics_names[0]} of {scores[0]}; {model.metrics_names[1]} of {scores[1] * 100}%')

model.summary()

scores  # [loss, accuracy] on test data...
model.metrics_names

# word_dict = {0: 'A', 1: 'B', 2: 'C', 3: 'D', 4: 'Delete', 5: 'E', 6: 'F', 7: 'G', 8: 'H', 9: 'I',
#              10: 'J', 11: 'K', 12: 'L', 13: 'M', 14: 'N', 15: 'nothing', 16: 'O', 17: 'P', 18: 'Q', 19: 'R',
#              20: 'S', 21: 'Spasi', 22: 'T', 23: 'U', 24: 'V', 25: 'W', 26: 'X', 27: 'Y', 28: 'Z'}
word_dict = {0: 'A', 1: 'B', 2: 'C', 3: 'D', 4: 'E', 5: 'F', 6: 'G', 7: 'H', 8: 'I', 9: 'J',
             10: 'K', 11: 'L', 12: 'M', 13: 'N', 14: 'O', 15: 'P', 16: 'Q', 17: 'R', 18: 'S', 19: 'T',
             20: 'U', 21: 'V', 22: 'W', 23: 'X', 24: 'Y', 25: 'Z'}
# word_dict = {0: '0', 1: '1', 2: '2', 3: '3', 4: '4', 5: '5', 6: '6', 7: '7', 8: '8', 9: '9',
#              10: 'A', 11: 'B', 12: 'C', 13: 'D', 14: 'E', 15: 'F', 16: 'G', 17: 'H', 18: 'I', 19: 'J',
#              20: 'K', 21: 'L', 22: 'M', 23: 'N', 24: 'O', 25: 'P', 26: 'Q', 27: 'R', 28: 'S', 29: 'T',
#              30: 'U', 31: 'V', 32: 'W', 33: 'X', 34: 'Y', 35: 'Z'}

predictions = model.predict(imgs, verbose=0)
print("predictions on a small set of test data--")
print("")
for ind, i in enumerate(predictions):
    print(word_dict[np.argmax(i)], end='   ')

# plotImages(imgs)
print('Actual labels')
for i in labels:
    print(word_dict[np.argmax(i)], end='   ')

print(imgs.shape)

history2.history

# plt.figure()
# plt.plot(history2.history['loss'])
# plt.plot(history2.history['val_loss'])
# plt.xlabel('epochs')
# plt.ylabel('Loss')
# plt.legend(['train_loss','val_loss'], loc=0)
# plt.show()
# plt.savefig('output/Loss.png')
#
# plt.plot(history2.history['accuracy'])
# plt.plot(history2.history['val_accuracy'])
# plt.xlabel('epochs')
# plt.ylabel('Accuracy')
# plt.legend(['train_accuracy','val_accuracy'], loc=0)
# plt.show()
# plt.savefig('output/Accuracy.png')