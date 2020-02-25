from __future__ import absolute_import, division, print_function, unicode_literals
from pathlib import Path
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Conv2D, Flatten, Dropout, MaxPooling2D
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.utils import to_categorical
import os
import numpy as np
# import matplotlib
# matplotlib.use('Agg')
gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
    try:
        # Currently, memory growth needs to be the same across GPUs
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
        logical_gpus = tf.config.experimental.list_logical_devices('GPU')
        print(len(gpus), "Physical GPUs,", len(logical_gpus), "Logical GPUs")
    except RuntimeError as e:
        # Memory growth must be set before GPUs have been initialized
        print(e)
print("Num GPUs Available: ", len(
    tf.config.experimental.list_physical_devices('GPU')))

pwd = os.path.dirname(os.path.realpath(__file__))
# ~/AI-Portfolio/ImageClassification/
PATH = os.path.join(pwd, 'piecesDatasetAll')

train_dir = Path(os.path.join(PATH, 'train'))
test_dir = Path(os.path.join(PATH, 'test'))

total_train_size = 0
train_sizes = {}
for child in train_dir.iterdir():
    file_amount = len(list(child.glob('*.jpeg')))
    print(f'Train - Size of {child.name} = {file_amount}')
    train_sizes[child.name] = file_amount
    total_train_size += file_amount

total_test_size = 0
test_sizes = {}
for child in test_dir.iterdir():
    file_amount = len(list(child.glob('*.jpeg')))
    print(f'Test - Size of {child.name} = {file_amount}')
    test_sizes[child.name] = file_amount
    total_test_size += file_amount

print(
    f'Training data size: {total_train_size}. Test data size {total_test_size}')
labels = [x for x in train_sizes.keys()]
CLASS_NAMES = np.array(sorted(labels))
print(CLASS_NAMES)

batch_size = 256
epochs = 15
IMG_HEIGHT = 50
IMG_WIDTH = 50

train_image_generator = ImageDataGenerator(rescale=1./255,
                                           validation_split=0.2)
test_image_generator = ImageDataGenerator(rescale=1./255)

train_data_gen = train_image_generator.flow_from_directory(
    batch_size=batch_size,
    directory=train_dir,
    shuffle=True,
    target_size=(IMG_HEIGHT, IMG_WIDTH),
    class_mode='binary',
    subset='training')

val_data_gen = train_image_generator.flow_from_directory(
    batch_size=batch_size,
    directory=train_dir,
    shuffle=True,
    target_size=(IMG_HEIGHT, IMG_WIDTH),
    class_mode='binary',
    subset='validation')

test_data_gen = test_image_generator.flow_from_directory(
    batch_size=batch_size,
    directory=test_dir,
    shuffle=True,
    target_size=(IMG_HEIGHT, IMG_WIDTH),
    class_mode='binary')

model = Sequential([
    Conv2D(32, 3, padding='same', activation='relu',
           input_shape=(IMG_HEIGHT, IMG_WIDTH, 3)),
    MaxPooling2D(),
    Conv2D(64, 3, padding='same', activation='relu'),
    MaxPooling2D(),
    Dropout(0.2),
    Flatten(),
    Dense(1024, activation='relu'),
    Dense(13, activation='softmax')
])

model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

model.summary()

checkpoint_path = os.path.join(
    pwd, "checkpoints", 'weights.{epoch:02d}-{val_loss:.2f}.h5')
checkpoint_dir = os.path.dirname(checkpoint_path)

cp_callback = tf.keras.callbacks.ModelCheckpoint(filepath=checkpoint_path,
                                                 save_weights_only=False,
                                                 verbose=1)

history = model.fit_generator(train_data_gen,
                              steps_per_epoch=total_train_size // batch_size,
                              epochs=epochs,
                              validation_data=val_data_gen,
                              validation_steps=total_test_size // batch_size,
                              use_multiprocessing=True,
                              callbacks=[cp_callback]
                              )

model.save(os.path.join(pwd, '13d_256b_15e.h5'))

acc = history.history['accuracy']
val_acc = history.history['val_accuracy']

loss = history.history['loss']
val_loss = history.history['val_loss']

unseen_loss, unseen_acc = model.evaluate_generator(test_data_gen)
print(f'New data - loss: {unseen_loss}, accuracy: {unseen_acc}')
epochs_range = range(epochs)

plt.figure(figsize=(8, 8))
plt.subplot(1, 2, 1)
plt.plot(epochs_range, acc, label='Training Accuracy')
plt.plot(epochs_range, val_acc, label='Validation Accuracy')
plt.legend(loc='lower right')
plt.title('Training and Validation Accuracy')

plt.subplot(1, 2, 2)
plt.plot(epochs_range, loss, label='Training Loss')
plt.plot(epochs_range, val_loss, label='Validation Loss')
plt.legend(loc='upper right')
plt.title('Training and Validation Loss')
plt.show()
