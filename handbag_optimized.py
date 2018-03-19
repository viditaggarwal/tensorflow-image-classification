from keras.preprocessing.image import ImageDataGenerator
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D
from keras.layers import Activation, Dropout, Flatten, Dense
from keras import backend as K
import numpy as np
import sys

# dimensions of our images.
img_width, img_height = 128, 128

train_data_dir = 'data_handbags_2/train'
validation_data_dir = 'data_handbags_2/validation'
nb_train_samples = 152
nb_validation_samples = 17
epochs = 12
batch_size = 16

if K.image_data_format() == 'channels_first':
    input_shape = (3, img_width, img_height)
else:
    input_shape = (img_width, img_height, 3)

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

# this is the augmentation configuration we will use for training
train_datagen = ImageDataGenerator(
    rescale=1. / 255,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=False)

# this is the augmentation configuration we will use for testing:
# only rescaling
test_datagen = ImageDataGenerator(rescale=1. / 255)

train_generator = train_datagen.flow_from_directory(
    train_data_dir,
    target_size=(img_width, img_height),
    batch_size=batch_size,
    class_mode='binary')

validation_generator = test_datagen.flow_from_directory(
    validation_data_dir,
    target_size=(img_width, img_height),
    batch_size=batch_size,
    class_mode='binary')

test_generator = test_datagen.flow_from_directory(
    'data_handbags_2/test',
    target_size=(img_width, img_height),
    batch_size=batch_size,
    class_mode='binary')

model.fit_generator(
    train_generator,
    steps_per_epoch=nb_train_samples // batch_size,
    epochs=epochs,
    validation_data=validation_generator,
    validation_steps=nb_validation_samples // batch_size)

model.save_weights(sys.argv[1])
scoreSeg = model.evaluate_generator(test_generator)
print("Accuracy = ",scoreSeg[1])

# model.load_weights('optimized_result_20_epochs.h5')

# train_generator = train_datagen.flow_from_directory(
#         'data_handbags/train',
#         target_size=(150, 150),
#         batch_size=batch_size,
#         class_mode=None,  # this means our generator will only yield batches of data, no labels
#         shuffle=False)  # our data will be in order, so all first 1000 images will be cats, then 1000 dogs
# # the predict_generator method returns the output of a model, given
# # a generator that yields batches of numpy data
# bottleneck_features_train = model.predict_generator(generator, nb_train_samples)
# # save the output as a Numpy array
# np.save(open('bottleneck_features_train.npy', 'w'), bottleneck_features_train)

# validation_generator = test_datagen.flow_from_directory(
#         'data_handbags/validation',
#         target_size=(150, 150),
#         batch_size=batch_size,
#         class_mode=None,
#         shuffle=False)
# bottleneck_features_validation = model.predict_generator(generator, nb_validation_samples)
# np.save(open('bottleneck_features_validation.npy', 'w'), bottleneck_features_validation)


# train_data = np.load(open('bottleneck_features_train.npy'))
# # the features were saved in order, so recreating the labels is easy
# train_labels = np.array([0] * 88 + [1] * 64)

# validation_data = np.load(open('bottleneck_features_validation.npy'))
# validation_labels = np.array([0] * 9 + [1] * 8)

# model = Sequential()
# # model.add(Flatten(input_shape=))
# model.add(Flatten(input_shape=(150,150,3)))
# model.add(Dense(128, activation='relu'))
# model.add(Dropout(0.5))
# model.add(Dense(1, activation='sigmoid'))

# model.compile(optimizer='rmsprop',
#               loss='binary_crossentropy',
#               metrics=['accuracy'])

# model.fit_generator(train_generator,
# 		steps_per_epoch=nb_train_samples // batch_size,
# 		epochs=epochs,
# 		validation_data=validation_generator,
# 		validation_steps=nb_validation_samples // batch_size)
# model.save_weights('bottleneck_fc_model.h5')
# scoreSeg = model.evaluate_generator(test_generator)
# print("Accuracy = ",scoreSeg[1])


