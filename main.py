import tensorflow as tf
import tensorflow_datasets as tfds
import numpy as np
from tensorflow import keras
from tensorflow.data.experimental import AUTOTUNE

global WIDTH, HEIGHT, CHANNELS, CLASSES, KERNEL_SIZE

WIDTH, HEIGHT, CHANNELS, KERNEL_SIZE = 128, 128, 3, 3

EPOCHS, BATCH_SIZE = 200, 65

data_splits, info = tfds.load('oxford_flowers102', split=['test', 'validation', 'train'], with_info=True)

(train, valid, test) = data_splits

CLASSES = info.features['label'].num_classes

# tfds.visualization.show_examples(train, info)


def parser(tensor):
    image = tensor['image']
    image = tf.image.resize(image, (WIDTH, HEIGHT)) / 255.0
    label = tensor['label']
    label = tf.one_hot(tensor['label'], 102)
    return image, label

def create_model():
    filter_size = WIDTH - KERNEL_SIZE + 1

    adam = keras.optimizers.Adam(learning_rate=0.001)

    augs = keras.Sequential([
        keras.layers.RandomFlip(),
        keras.layers.RandomTranslation(0.2, 0.2),
        keras.layers.RandomRotation((-0.1, 0.1)),
        keras.layers.RandomContrast(0.1),
    ])

    # Create model
    model = keras.Sequential()
    #   Define input shape
    model.add(keras.layers.Input(shape=(WIDTH, HEIGHT) + (CHANNELS,)))
    #   Add augmentations, only runs when calling fit(), not on evaluate() or predict()
    model.add(augs)
    
    #   Convolutions, followed by Pooling and Dropout
    model.add(keras.layers.Conv2D(filter_size, (KERNEL_SIZE, KERNEL_SIZE), activation='relu', padding='same'))
    model.add(keras.layers.MaxPooling2D())
    model.add(keras.layers.Dropout(0.2)) 
    
    #   Reduce filter size to adjust for shape changing
    filter_size -= (KERNEL_SIZE - 1)

    #   Repeat
    model.add(keras.layers.Conv2D(filter_size, (KERNEL_SIZE, KERNEL_SIZE), activation='relu', padding='same'))
    model.add(keras.layers.MaxPooling2D())
    model.add(keras.layers.Dropout(0.2)) 
    
    filter_size -= (KERNEL_SIZE - 1)

    model.add(keras.layers.Conv2D(filter_size, (KERNEL_SIZE, KERNEL_SIZE), activation='relu', padding='same'))
    model.add(keras.layers.MaxPooling2D())
    model.add(keras.layers.Dropout(0.2))

    filter_size -= (KERNEL_SIZE - 1)
    
    model.add(keras.layers.Conv2D(filter_size, (KERNEL_SIZE, KERNEL_SIZE), activation='relu', padding='same'))
    model.add(keras.layers.MaxPooling2D())
    model.add(keras.layers.Dropout(0.2)) 
    
    # Flatten data to fit to Dense layer
    model.add(keras.layers.Flatten())
    
    # Dense layer of size CLASSES * 2n, for n >= 0
    model.add(keras.layers.Dense(CLASSES * 2, activation='relu'))
    model.add(keras.layers.Dense(CLASSES, activation='softmax'))
    
    #   Compile the model with defined optimiser, appropriate loss function for one-hot labels
    model.compile(optimizer=adam, loss='categorical_crossentropy', metrics=['accuracy'])

    return model

# Shuffle data, then map our parser function over the data splits, parallelisation, define batches
train = train.shuffle(32).map(parser, num_parallel_calls=AUTOTUNE).batch(BATCH_SIZE).prefetch(AUTOTUNE)

valid = valid.shuffle(32).map(parser, num_parallel_calls=AUTOTUNE).batch(BATCH_SIZE).prefetch(AUTOTUNE)

test = test.shuffle(32).map(parser, num_parallel_calls=AUTOTUNE).batch(BATCH_SIZE).prefetch(AUTOTUNE)

# Create model
model = create_model()

# Output model structure
print(model.summary())

# Training
model.fit(train, validation_data=valid, epochs=EPOCHS)

# Give accuracies for validation set and testing set
v_scores = model.evaluate(valid, verbose=0)
t_scores = model.evaluate(test, verbose=0)
print("Validation set accuracy: %.2f%%" % (v_scores[1]*100))
print("Testing set accuracy: %.2f%%" % (t_scores[1]*100))
