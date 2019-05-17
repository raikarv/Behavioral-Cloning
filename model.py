import os
import csv
import cv2
import numpy as np
from scipy import ndimage
from sklearn.utils import shuffle
from sklearn.model_selection import train_test_split
from keras.optimizers import Adam
from keras.models import Sequential
from keras.layers import Convolution2D, Dropout, MaxPooling2D, Flatten, Activation, Dense, Cropping2D, Lambda
#from keras import backend as K
#backend.set_image_dim_ordering('tf')


lines = []
currentPath = './data/IMG/'
with open('./data/driving_log.csv') as csvfile:
    reader = csv.reader(csvfile)
    for line in reader:
        lines.append(line)
del(lines[0])

#Function to load images and resize
def processImage(fileName):
    image = ndimage.imread(fileName)
    #image = cv2.resize(image[65:135,:], (64,64))
    return image

def resize(image):
    import cv2
    from keras.backend import tf as ktf   
    resized = ktf.image.resize_images(image, (64, 64))
    return resized

def generateData(samples, batchSize = 32, leftAngOff = 0.4, rightAngOff = -0.3):
    numSamples = len(samples)
    while True: # Loop forever so the generator never terminates
        samples = shuffle(samples)
        for offset in range(0, numSamples, batchSize):
            images = []
            angles = []
            batchSamples = samples[offset: offset + batchSize]
            for line in batchSamples:
                #Center Image
                image = processImage(currentPath + line[0].split('/')[-1])
                steeringAngle = float(line[3])
                images.append(image)
                angles.append(steeringAngle)
                
                #Flip center images, angles and add to data set
                images.append(np.fliplr(image))
                angles.append(-steeringAngle)
                
                #Add left camera image and steering angle with offset
                images.append(processImage(currentPath + line[1].split('/')[-1]))
                angles.append(steeringAngle + leftAngOff)
                
                #Add right camera image and steering angle with offset
                images.append(processImage(currentPath + line[2].split('/')[-1]))
                angles.append(steeringAngle + rightAngOff)
            X_train = np.array(images)
            y_train = np.array(angles)
            yield shuffle(X_train, y_train)
 
# Split training and validation set
train, valid = train_test_split(lines, test_size = 0.2)
print('Training Set = ', len(train), '\nValidation Set = ', len(valid))

#Define Model, based on NVIDIA's Self -Driving Cars Paper
model = Sequential()
model.add(Cropping2D(cropping=((75,25), (0,0)), input_shape=(160,320,3), data_format="channels_last"))
model.add(Lambda(resize))
model.add(Lambda(lambda x: x/127.5 - 1, input_shape=(64, 64, 3)))
model.add(Convolution2D(24, 5, 5, border_mode='same', subsample=(2,2)))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2,2), strides = (1,1)))
model.add(Convolution2D(36, 5, 5, border_mode='same', subsample=(2,2)))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2,2), strides = (1,1)))
model.add(Convolution2D(48, 5, 5, border_mode='same', subsample=(2,2)))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2,2), strides = (1,1)))
model.add(Convolution2D(64, 3, 3, border_mode='same', subsample=(1,1)))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2,2), strides = (1,1)))
model.add(Flatten())

#Five fully connected layers
model.add(Dense(1164, activation='relu'))
model.add(Dense(100, activation='relu'))
model.add(Dense(50, activation='relu'))
model.add(Dense(10, activation='relu'))
model.add(Dense(1))

learnRate = 1e-4
adam = Adam(learnRate)
model.compile(optimizer = adam, loss='mse')

model.summary()

epochs = 5
batchSize = 32
epoch_samples = len(train)/batchSize
validation_samples = len(valid)
print('EPOCHS = ', epochs, '\nBatch Size = ', batchSize, '\nSamples/EPOCHS =', epoch_samples)

history = model.fit_generator(generateData(train), samples_per_epoch = epoch_samples,
                              nb_epoch = epochs, validation_data = generateData(valid),
                              nb_val_samples = validation_samples, verbose = 1)



#Save Model
#print(train[5].shape, valid[5].shape)
model.save('model.h5')


