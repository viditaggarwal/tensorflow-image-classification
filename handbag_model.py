from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D
from keras.layers import Activation, Dropout, Flatten, Dense
from keras.preprocessing.image import ImageDataGenerator
from keras.preprocessing.image import img_to_array
import numpy as np
import argparse
import cv2
import sys, os
from PIL import Image # $ pip install pillow
import urllib

batch_size = 16

# this is the augmentation configuration we will use for training
train_datagen = ImageDataGenerator(
        rescale=1./255,
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True)

# this is the augmentation configuration we will use for testing:
# only rescaling
test_datagen = ImageDataGenerator(rescale=1./255)
def model_three_cnn_700():
	img_width, img_height = 128, 128
	model = Sequential()
	model.add(Conv2D(32, (3, 3), input_shape=(img_width, img_height, 3)))
	model.add(Activation('relu'))
	model.add(MaxPooling2D(pool_size=(2, 2), dim_ordering="th", padding='same'))

	# model.add(Conv2D(32, (3, 3)))
	# model.add(Activation('relu'))
	# model.add(MaxPooling2D(pool_size=(2, 2), dim_ordering="th", padding='same'))

	model.add(Conv2D(64, (3, 3)))
	model.add(Activation('relu'))
	model.add(MaxPooling2D(pool_size=(2, 2), dim_ordering="th", padding='same'))

	model.add(Conv2D(128, (3, 3)))
	model.add(Activation('relu'))
	model.add(MaxPooling2D(pool_size=(2, 2), dim_ordering="th", padding='same'))

	model.add(Flatten())  # this converts our 3D feature maps to 1D feature vectors
	model.add(Dense(128))
	model.add(Activation('relu'))
	model.add(Dropout(0.5))
	model.add(Dense(1))
	model.add(Activation('sigmoid'))

	model.compile(loss='binary_crossentropy',
	              optimizer='rmsprop',
	              metrics=['accuracy'])
	return model, 128, 128


def model_four_cnn():
	model = Sequential()
	model.add(Conv2D(32, (3, 3), input_shape=(128, 128, 3)))
	model.add(Activation('relu'))
	model.add(MaxPooling2D(pool_size=(2, 2), dim_ordering="th", padding='same'))

	model.add(Conv2D(32, (3, 3)))
	model.add(Activation('relu'))
	model.add(MaxPooling2D(pool_size=(2, 2), dim_ordering="th", padding='same'))

	model.add(Conv2D(64, (3, 3)))
	model.add(Activation('relu'))
	model.add(MaxPooling2D(pool_size=(2, 2), dim_ordering="th", padding='same'))

	model.add(Conv2D(128, (3, 3)))
	model.add(Activation('relu'))
	model.add(MaxPooling2D(pool_size=(2, 2), dim_ordering="th", padding='same'))

	model.add(Flatten())  # this converts our 3D feature maps to 1D feature vectors
	model.add(Dense(128))
	model.add(Activation('relu'))
	model.add(Dropout(0.5))
	model.add(Dense(1))
	model.add(Activation('sigmoid'))

	model.compile(loss='binary_crossentropy',
	              optimizer='rmsprop',
	              metrics=['accuracy'])
	return model, 128, 128

def model_one_cnn():
	model = Sequential()
	model.add(Conv2D(32, (3, 3), input_shape=(150, 150, 3)))
	model.add(Activation('relu'))
	model.add(MaxPooling2D(pool_size=(2, 2), dim_ordering="th", padding='same'))

	model.add(Flatten())  # this converts our 3D feature maps to 1D feature vectors
	model.add(Dense(64))
	model.add(Activation('relu'))
	model.add(Dropout(0.5))
	model.add(Dense(1))
	model.add(Activation('sigmoid'))

	model.compile(loss='binary_crossentropy',
              optimizer='rmsprop',
              metrics=['accuracy'])
	return model, 150, 150

def model_two_cnn():
	model = Sequential()
	model.add(Conv2D(32, (3, 3), input_shape=(150, 150, 3)))
	model.add(Activation('relu'))
	model.add(MaxPooling2D(pool_size=(2, 2), dim_ordering="th", padding='same'))

	model.add(Conv2D(32, (3, 3)))
	model.add(Activation('relu'))
	model.add(MaxPooling2D(pool_size=(2, 2), dim_ordering="th", padding='same'))

	model.add(Flatten())  # this converts our 3D feature maps to 1D feature vectors
	model.add(Dense(64))
	model.add(Activation('relu'))
	model.add(Dropout(0.5))
	model.add(Dense(1))
	model.add(Activation('sigmoid'))

	model.compile(loss='binary_crossentropy',
              optimizer='rmsprop',
              metrics=['accuracy'])
	return model, 150, 150

def model_three_cnn():
	model = Sequential()
	model.add(Conv2D(32, (3, 3), input_shape=(150, 150, 3)))
	model.add(Activation('relu'))
	model.add(MaxPooling2D(pool_size=(2, 2), dim_ordering="th", padding='same'))

	model.add(Conv2D(32, (3, 3)))
	model.add(Activation('relu'))
	model.add(MaxPooling2D(pool_size=(2, 2), dim_ordering="th", padding='same'))

	model.add(Conv2D(64, (3, 3)))
	model.add(Activation('relu'))
	model.add(MaxPooling2D(pool_size=(2, 2), dim_ordering="th", padding='same'))

	model.add(Flatten())  # this converts our 3D feature maps to 1D feature vectors
	model.add(Dense(64))
	model.add(Activation('relu'))
	model.add(Dropout(0.5))
	model.add(Dense(1))
	model.add(Activation('sigmoid'))

	model.compile(loss='binary_crossentropy',
              optimizer='rmsprop',
              metrics=['accuracy'])
	return model, 150, 150

def get_model(model_name):
	if 'one' in model_name:
		return model_one_cnn()
	elif 'two' in model_name:
		return model_two_cnn()
	elif '700' in model_name and 'three' in model_name:
		return model_three_cnn_700()
	elif 'three' in model_name:
		return model_three_cnn()
	elif 'four' in model_name:
		return model_four_cnn()
	return None

def train_model(model_name):
	# this is a generator that will read pictures found in
	# subfolers of 'data/train', and indefinitely generate
	# batches of augmented image data
	train_generator = train_datagen.flow_from_directory(
	        'data_handbags/train',  # this is the target directory
	        target_size=(150, 150),  # all images will be resized to 150x150
	        batch_size=batch_size,
	        class_mode='binary')  # since we use binary_crossentropy loss, we need binary labels

	# this is a similar generator, for validation data
	validation_generator = test_datagen.flow_from_directory(
	        'data_handbags/validation',
	        target_size=(150, 150),
	        batch_size=batch_size,
	        class_mode='binary')

	test_generator = test_datagen.flow_from_directory(
	        'data_handbags/test',
	        target_size=(150, 150),
	        batch_size=batch_size,
	        class_mode='binary')

	model = get_model(model_name)
	model.fit_generator(
	        train_generator,
	        steps_per_epoch=2000 // batch_size,
	        epochs=3,
	        validation_data=validation_generator,
	        validation_steps=800 // batch_size)
	
	model.save_weights(model_name)  # always save your weights after training or during training
	scoreSeg = model.evaluate_generator(test_generator)
	print("Accuracy = ",scoreSeg[1])

def test_input(model_name):
	model = get_model(model_name)
	model.load_weights(model_name)
	for filename in os.listdir(sys.argv[3]):
		print str(sys.argv[3]) + '/' + str(filename)
		try:
			image = cv2.imread(str(sys.argv[3]) + '/' + str(filename))
			orig = image.copy()
			 
			# pre-process the image for classification
			image = cv2.resize(image, (150, 150))
			image = image.astype("float") / 255.0
			image = img_to_array(image)
			image = np.expand_dims(image, axis=0)
			print str(filename) + " : " + str(model.predict(image)[0])
		except:
			print "Exception : " + str(filename)

def test_url(model_name):
	url = sys.argv[3]
	model = get_model(model_name)
	model.load_weights(model_name)
	im = Image.open(urllib.urlopen(url))
	image = cv2.cvtColor(np.asarray(im), cv2.COLOR_RGB2BGR)
	orig = image.copy()
	 
	# pre-process the image for classification
	image = cv2.resize(image, (150, 150))
	image = image.astype("float") / 255.0
	image = img_to_array(image)
	image = np.expand_dims(image, axis=0)
	print "Score: " + str(model.predict(image)[0])


def test_image_url(model_name, url):
	model, img_height, img_width = get_model(model_name)
	model.load_weights(model_name)
	im = Image.open(urllib.urlopen(url))
	image = cv2.cvtColor(np.asarray(im), cv2.COLOR_RGB2BGR)
	orig = image.copy()
	 
	# pre-process the image for classification
	image = cv2.resize(image, (img_height, img_width))
	image = image.astype("float") / 255.0
	image = img_to_array(image)
	image = np.expand_dims(image, axis=0)
	result = {}
	result["Score"] = str(model.predict(image)[0][0])
	return result


if __name__ == "__main__":
	if sys.argv[1] == 'test':
		test_input(sys.argv[2])
	elif sys.argv[1] == 'train':
		train_model(sys.argv[2])
	elif sys.argv[1] == 'url':
		test_url(sys.argv[2])







