import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from keras import Input, layers

import os    
os.chdir(os.path.dirname(os.path.abspath(__file__)))

from zipfile import ZipFile
data_path = 'animal_data.zip'
with ZipFile(data_path, 'r') as zip:
    zip.extractall()
    print('The data set has been extracted.')

path = 'animals'
classes = os.listdir(path)
print(classes)

model = tf.keras.models.load_model('proj1_model.h5')
model.summary()

batch_size = 100
img_height = 96
img_width = 96

#test_img = tf.keras.utils.image_dataset_from_directory(
#	path,
#	validation_split=0.2,
#	subset="testing",
#	seed=123,
#	image_size=(img_height, img_width),
#	batch_size=batch_size)

loss, acc = model.evaluate(test_img)

print(f"Accuracy: {acc}")