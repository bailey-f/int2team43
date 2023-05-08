import tensorflow as tf
import tensorflow_datasets as tfds
import numpy as np
from tensorflow import keras
from tensorflow.data.experimental import AUTOTUNE

global WIDTH, HEIGHT, CHANNELS, CLASSES

WIDTH, HEIGHT, CHANNELS = 128, 128, 3

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
    adam = keras.optimizers.Adam(learning_rate=0.001)

    model = keras.Sequential()
    # model.add(keras.layers.Input(shape=(WIDTH, HEIGHT) + (CHANNELS,)))

    model.add(keras.layers.Conv2D(126, (3, 3), input_shape=(WIDTH, HEIGHT) + (CHANNELS,), padding='same'))
    model.add(keras.layers.MaxPooling2D())
    

    model.add(keras.layers.Conv2D(64, (3, 3), padding='same'))
    model.add(keras.layers.MaxPooling2D())
   

    model.add(keras.layers.Conv2D(64, (3, 3), padding='same'))
    model.add(keras.layers.MaxPooling2D())
    model.add(keras.layers.Dropout(0.2))

    model.add(keras.layers.Flatten())

    model.add(keras.layers.Dense(102, activation='relu'))
    model.add(keras.layers.Dense(CLASSES, activation='softmax'))

    model.compile(optimizer=adam, loss='categorical_crossentropy', metrics=['accuracy'])

    return model

train = train.shuffle(32).map(parser, num_parallel_calls=AUTOTUNE).batch(65).prefetch(AUTOTUNE)

valid = valid.shuffle(32).map(parser, num_parallel_calls=AUTOTUNE).batch(65).prefetch(AUTOTUNE)

test = test.shuffle(32).map(parser, num_parallel_calls=AUTOTUNE).batch(65).prefetch(AUTOTUNE)

model = create_model()

print(model.summary())

model.fit(train, validation_data=valid, epochs=25)

v_scores = model.evaluate(valid, verbose=0)
t_scores = model.evaluate(test, verbose=0)
print("Validation set accuracy: %.2f%%" % (v_scores[1]*100))
print("Testing set accuracy: %.2f%%" % (t_scores[1]*100))
