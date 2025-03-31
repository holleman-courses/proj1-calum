import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
#from tensorflow.keras import Input, layers
import keras
from keras import Input, layers
from keras import ops

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


batch_size = 100
img_height = 96
img_width = 96

train_img = tf.keras.utils.image_dataset_from_directory(
  	path,
  	validation_split=0.2,
  	subset="training",
  	seed=123,
  	image_size=(img_height, img_width),
  	batch_size=batch_size)

val_img = tf.keras.utils.image_dataset_from_directory(
	path,
	validation_split=0.2,
	subset="validation",
	seed=123,
	image_size=(img_height, img_width),
	batch_size=batch_size)

#val_frac = 0.1
#num_val_samples = int(len(train_img)*val_frac)
# choose num_val_samples indices up to the size of train_images, !replace => no repeats
#val_idxs = np.random.choice(np.arange(len(train_img)), size=num_val_samples, replace=False)
#trn_idxs = np.setdiff1d(np.arange(len(train_img)), val_idxs)
#val_images = train_img[val_idxs, :,:,:]
#train_images = train_img[trn_idxs, :,:,:]

#val_labels = classes[val_idxs]
#train_labels = classes[trn_idxs]

#train_labels = train_labels.squeeze()
#test_labels = test_labels.squeeze()
#val_labels = val_labels.squeeze()

#input_shape = train_img.shape[1:]
#train_images = train_img / 255.0
#test_images = test_images  / 255.0
#val_images = val_img   / 255.0


def skip_block(x, filter1, filter2, stride1=1, stride2=1):
	skip = layers.Conv2D(filter1, kernel_size=(3,3),
						strides=(stride1,stride1), padding='same')(x)
	skip = layers.BatchNormalization()(skip)
	skip = layers.Activation('relu')(skip)

	skip = layers.Conv2D(filter2, kernel_size=(3,3),
						strides=(stride2,stride2), padding='same')(skip)
	skip = layers.BatchNormalization()(skip)
	skip = layers.Activation('relu')(skip)

	oneDconv = layers.Conv2D(filter2, kernel_size=(1,1),
							strides=(stride1*stride2,stride1*stride2),
							padding='same')(x)
	oneDconv = layers.BatchNormalization()(oneDconv)

	out = layers.Add()([skip, oneDconv])
	out = layers.Activation('relu')(out)
	return out

def build_model():

	inputs = tf.keras.Input(shape=(img_height, img_width, 3))
	conv = layers.Conv2D(32, kernel_size=(3,3), strides=(2,2), padding='same') #conv1
	x = conv(inputs)
	x = layers.BatchNormalization()(x)
	x = layers.Activation('relu')(x)

	#x = layers.Conv2D(64, kernel_size=(3,3), strides=(2,2), padding='same')(x) #conv2
	#x = layers.BatchNormalization()(x)
	#x = layers.Activation('relu')(x)

	#x = layers.Conv2D(128, kernel_size=(3,3), strides=(2,2), padding='same')(x) #conv3
	#x = layers.BatchNormalization()(x)
	#x = layers.Activation('relu')(x)

	x = skip_block(x,64,128,2,2)	#conv 2,3
	x = skip_block(x,128,128)		#conv 4,5
	x = skip_block(x,128,128)		#conv 6,7

	x = layers.MaxPooling2D(pool_size=(4, 4), strides=(4,4))(x)
	x = layers.Flatten()(x)
	x = layers.Dense(128)(x)
	x = layers.BatchNormalization()(x)
	x = layers.Activation('relu')(x)

	outputs = layers.Dense(10)(x)
	model = tf.keras.Model(inputs=inputs, outputs=outputs, name="model")

	return model

## Build and train model 1
model = build_model()

model.compile(optimizer='adam',
			loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
			metrics=['accuracy'])
	
model.summary()

#train_model = model.fit(train_img, train_labels, 
#               validation_data=(val_img, val_labels),
#               epochs=1)

train_model = model.fit(
				train_img,
            	validation_data=val_img,
            	epochs=5)

	## model1: acc: ---, loss: ---, val_acc: ---, val_loss: ---
	## Total params: --- (--- MB)

loss = model.evaluate(train_img)
print("Final training loss:", loss)

plt.plot(train_model.history['accuracy'])
plt.plot(train_model.history['val_accuracy'])
plt.title('model accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train', 'val'], loc='upper left')
#plt.show()
plt.savefig('modelacc.png')
plt.close()

plt.plot(train_model.history['loss'])
plt.plot(train_model.history['val_loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'val'], loc='upper left')
#plt.show()
plt.savefig('modelloss.png')

model.save("proj1_model.h5")
