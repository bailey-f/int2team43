import tensorflow as tf
import numpy as np
from tensorflow import keras
from keras.utils import np_utils
from scipy.io import loadmat

def create_model(X_train, y_train, X_valid, y_valid):
    shape = X_train.shape
    
    # Creating the model
    model = tf.keras.Sequential()
    
    #   Convolution layer
    model.add(keras.layers.Conv2D(32, 3, input_shape=shape[1:], padding='same'))
    #   Activation
    model.add(keras.layers.Activation('relu'))
    #   Pooling
    model.add(keras.layers.MaxPooling2D(2))

    # Repeat
    model.add(keras.layers.Conv2D(64, 3, activation='relu', padding='same'))
    model.add(keras.layers.MaxPooling2D(2))

    model.add(keras.layers.Flatten())

    # Create densely connected layers
    model.add(keras.layers.Dense(128, activation='relu'))
    model.add(keras.layers.Dense(102))

    # Softmax activataion layer to select highest probability
    model.add(keras.layers.Dense(y_valid.shape[1], activation='softmax'))

    # Compile model
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

    return model

def preprocessing(image_set, labels, augment=False):
    img = []
    label = []
    WIDTH, HEIGHT, CHANNELS = 128, 128, 3
    for x in range(len(image_set)):
        # padding the start of the number with zeroes
        path = 'jpg/image_' + str(image_set[x]).zfill(5) + '.jpg'
        
        # Preprocessing to make image sizes uniform
        image = tf.io.read_file(path)
        image = tf.io.decode_jpeg(image)
        processed_image = tf.image.convert_image_dtype(image, tf.float32)
        processed_image = tf.image.resize(image, [WIDTH, HEIGHT])

        # Normalise the colour values
        processed_image = processed_image / 255.0
        
        img.append(processed_image)
        label.append(labels[image_set[x]-1])
        
        if augment:
            # random flips
            for y in range(3):
                seed = (y, 0)
                aug = tf.image.stateless_random_flip_left_right(processed_image, seed=seed)
                img.append(aug)
                label.append(labels[image_set[x]-1])
        
            # random jpeg quality
            for y in range(3):
                seed = (y, 0)
                aug = tf.image.stateless_random_jpeg_quality(processed_image, 50, 100, seed=seed)
                img.append(aug)
                label.append(labels[image_set[x]-1])

            # random crop
            for y in range(3):
                seed = (y, 0)
                aug = tf.image.stateless_random_crop(processed_image, size=[WIDTH, HEIGHT, CHANNELS], seed=seed)
                img.append(aug)
                label.append(labels[image_set[x]-1])

    img = np.asarray(img, dtype=np.float32)
    label = np.asarray(label, dtype=np.float32)
    return img, label

# Load datasplits
setid = loadmat('setid.mat')
setid.keys()
valid_id = setid['valid'][0]
test_id = setid['tstid'][0]
train_id = setid['trnid'][0]

# Load labels
labelfile = loadmat('imagelabels.mat')
label = labelfile['labels'][0]
for i in range(len(label)):
    label[i] = label[i] - 1

# Validation set
print("Preprocessing validation set")
X_valid, y_valid = preprocessing(valid_id, label)
y_valid = np_utils.to_categorical(y_valid)
print('Validation set preprocessed\n')

# Testing set
#print("Preprocessing testing set (This may take a minute)")
#X_test, y_test = preprocessing(test_id, label)
#y_test = np_utils.to_categorical(y_test)
#print('Testing set preprocessed\n')

# Training set
print("Preprocessing training set")
X_train, y_train = preprocessing(train_id, label, augment=True)
y_train = np_utils.to_categorical(y_train)
print('Training set preprocessed\n')

print(len(X_train))

model = create_model(X_train, y_train, X_valid, y_valid)

print(model.summary())

np.random.seed(42)
history = model.fit(X_train, y_train, validation_data=(X_valid, y_valid), epochs=25, batch_size=65)

scores = model.evaluate(X_valid, y_valid, verbose=0)
print("Accuracy: %.2f%%" % (scores[1]*100))
