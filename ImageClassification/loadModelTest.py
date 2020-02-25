from tensorflow.keras.preprocessing.image import ImageDataGenerator
import tensorflow as tf 
import os
from pathlib import Path

new_model = tf.keras.models.load_model('256b_15e.h5')

print(new_model.summary())

batch_size = 256
epochs = 15
IMG_HEIGHT = 50
IMG_WIDTH = 50

pwd = os.path.dirname(os.path.realpath(__file__))
PATH = os.path.join(pwd, 'piecesDataset')
test_dir = Path(os.path.join(PATH, 'test'))

test_image_generator = ImageDataGenerator(rescale=1./255)

test_data_gen = test_image_generator.flow_from_directory(
    batch_size=batch_size,
    directory=test_dir,
    shuffle=True,
    target_size=(IMG_HEIGHT, IMG_WIDTH),
    class_mode='binary')

unseen_loss, unseen_acc = new_model.evaluate_generator(test_data_gen)
print(f'New data - loss: {unseen_loss}, accuracy: {unseen_acc}')