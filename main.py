import tensorflow as tf
import tensorflow_datasets as tfds
import numpy as np
import matplotlib.pyplot as plt
from tensorflow import keras
from tensorflow.data.experimental import AUTOTUNE

global WIDTH, HEIGHT, CHANNELS, CLASSES, KERNEL_SIZE, NAMES

WIDTH, HEIGHT, CHANNELS, KERNEL_SIZE = 128, 128, 3, 3

EPOCHS, BATCH_SIZE = 1000, 65

data_splits, info = tfds.load('oxford_flowers102', split=['test', 'validation', 'train'], with_info=True)

(test, valid, train) = data_splits

CLASSES = info.features['label'].num_classes
NAMES = info.features['label'].names

mapping = {"21": "fire lily", "3": "canterbury bells", "45": "bolero deep blue", "1": "pink primrose", "34": "mexican aster", "27": "prince of wales feathers", "7": "moon orchid", "16": "globe-flower", "25": "grape hyacinth", "26": "corn poppy", "79": "toad lily", "39": "siam tulip", "24": "red ginger", "67": "spring crocus", "35": "alpine sea holly", "32": "garden phlox", "10": "globe thistle", "6": "tiger lily", "93": "ball moss", "33": "love in the mist", "9": "monkshood", "102": "blackberry lily", "14": "spear thistle", "19": "balloon flower", "100": "blanket flower", "13": "king protea", "49": "oxeye daisy", "15": "yellow iris", "61": "cautleya spicata", "31": "carnation", "64": "silverbush", "68": "bearded iris", "63": "black-eyed susan", "69": "windflower", "62": "japanese anemone", "20": "giant white arum lily", "38": "great masterwort", "4": "sweet pea", "86": "tree mallow", "101": "trumpet creeper", "42": "daffodil", "22": "pincushion flower", "2": "hard-leaved pocket orchid", "54": "sunflower", "66": "osteospermum", "70": "tree poppy", "85": "desert-rose", "99": "bromelia", "87": "magnolia", "5": "english marigold", "92": "bee balm", "28": "stemless gentian", "97": "mallow", "57": "gaura", "40": "lenten rose", "47": "marigold", "59": "orange dahlia", "48": "buttercup", "55": "pelargonium", "36": "ruby-lipped cattleya", "91": "hippeastrum", "29": "artichoke", "71": "gazania", "90": "canna lily", "18": "peruvian lily", "98": "mexican petunia", "8": "bird of paradise", "30": "sweet william", "17": "purple coneflower", "52": "wild pansy", "84": "columbine", "12": "colt's foot", "11": "snapdragon", "96": "camellia", "23": "fritillary", "50": "common dandelion", "44": "poinsettia", "53": "primula", "72": "azalea", "65": "californian poppy", "80": "anthurium", "76": "morning glory", "37": "cape flower", "56": "bishop of llandaff", "60": "pink-yellow dahlia", "82": "clematis", "58": "geranium", "75": "thorn apple", "41": "barbeton daisy", "95": "bougainvillea", "43": "sword lily", "83": "hibiscus", "78": "lotus lotus", "88": "cyclamen", "94": "foxglove", "81": "frangipani", "74": "rose", "89": "watercress", "73": "water lily", "46": "wallflower", "77": "passion flower", "51": "petunia"}


print("train: " + str(len(train)))
print("test: " + str(len(test)))
print("valid: " + str(len(valid)))

def parser(tensor):
    image = tensor['image']
    image = tf.image.resize(image, (WIDTH, HEIGHT)) / 255.0
    label = tensor['label']
    label = tf.one_hot(tensor['label'], 102)
    return image, label

def create_model():
    filter_size = WIDTH - KERNEL_SIZE + 1
    
    adam = keras.optimizers.Adam(learning_rate=0.0001)

    augs = keras.Sequential([
        keras.layers.RandomFlip("horizontal"),
        keras.layers.RandomTranslation((-0.1, 0.1), (-0.1, 0.1)),
        keras.layers.RandomRotation((-0.1, 0.1)),
        keras.layers.RandomContrast(0.1),
        keras.layers.RandomZoom((-0.2, 0), (-0.2, 0))
    ])

    # Create model
    model = keras.Sequential()
    
    #   Define input shape
    model.add(keras.layers.Input(shape=(WIDTH, HEIGHT) + (CHANNELS,)))
    #   Add augmentations, only runs when calling fit(), not on evaluate() or predict()
    model.add(augs)

    model.add(keras.layers.BatchNormalization())

    #   Block 1
    model.add(keras.layers.Conv2D(filter_size, (KERNEL_SIZE, KERNEL_SIZE), strides=2))
    model.add(keras.layers.Conv2D(filter_size, (KERNEL_SIZE+2, KERNEL_SIZE+2), strides=1))
    model.add(keras.layers.Conv2D(filter_size, (KERNEL_SIZE, KERNEL_SIZE), strides=1, padding='same'))
    
    model.add(keras.layers.Activation('relu'))

    model.add(keras.layers.BatchNormalization())

    model.add(keras.layers.MaxPooling2D(3))

    model.add(keras.layers.Dropout(0.2))

    #   Block 2
    model.add(keras.layers.Conv2D(filter_size, (KERNEL_SIZE, KERNEL_SIZE), strides=1))
    model.add(keras.layers.Conv2D(filter_size, (KERNEL_SIZE, KERNEL_SIZE), strides=2))
    model.add(keras.layers.Conv2D(filter_size, (KERNEL_SIZE, KERNEL_SIZE), strides=1))

    model.add(keras.layers.Activation('relu'))

    model.add(keras.layers.BatchNormalization())

    model.add(keras.layers.MaxPooling2D(3))

    model.add(keras.layers.Dropout(0.2))
    
    # Flatten data to fit to Dense layer
    model.add(keras.layers.Flatten())

    # Dense layer of size CLASSES * 2^n, for n >= 0
    model.add(keras.layers.Dense(CLASSES * 2, activation='relu'))
    model.add(keras.layers.Dense(CLASSES, activation='softmax'))
    
    #   Compile the model with defined optimiser, appropriate loss function for one-hot labels
    model.compile(optimizer=adam, loss='categorical_crossentropy', metrics=['accuracy'])

    return model

# train, save, and evaluate the model
def train_and_save(train, valid, test):

    # Create model
    model = create_model()

    # Output model structure
    print(model.summary())

    # Training
    model.fit(train, validation_data=valid, epochs=EPOCHS)
    
    # Save
    model.save('models/model.h5')
    print("Model saved to: 'models/model.h5'")

    # Give accuracies for testing set
    t_scores = model.evaluate(test, verbose=0)
    print("Testing set accuracy: %.2f%%" % (t_scores[1]*100))

def load_and_predict(image_path, actual_label, model_path):
    model = keras.models.load_model(model_path)

    image = tf.io.read_file(image_path)
    image = tf.io.decode_jpeg(image, channels=3)
    processed_image = tf.image.resize(image, (WIDTH, HEIGHT)) / 255.0
    processed_image = tf.convert_to_tensor(processed_image, dtype=tf.float32)
    
    # Expand image to batch where it is the only member
    processed_image = tf.reshape(processed_image, (-1, WIDTH, HEIGHT, 3))

    predictions = model.predict(processed_image)
    top_10 = np.argpartition(predictions[0], -10)[-10:]
    prediction = np.argmax(predictions[0])
    
    plt.imshow(image, cmap=plt.cm.binary)

    plt.xlabel("{} {:2.0f}% ({})".format(mapping[str(prediction + 1)],
                                         100*np.max(predictions),
                                         actual_label,
                                         color='blue'))

    
    for i in top_10:
        print(mapping[str(i + 1)] + " | %.2f%%" % (100 * predictions[0][i]))

    plt.show()


# Shuffle data, then map our parser function over the data splits, parallelisation, define batches
train = train.shuffle(320).map(parser, num_parallel_calls=AUTOTUNE).batch(BATCH_SIZE).prefetch(AUTOTUNE)

valid = valid.shuffle(320).map(parser, num_parallel_calls=AUTOTUNE).batch(BATCH_SIZE).prefetch(AUTOTUNE)

test = test.shuffle(320).map(parser, num_parallel_calls=AUTOTUNE).batch(BATCH_SIZE).prefetch(AUTOTUNE)

# UNCOMMENT TO TRAIN

# train_and_save(train, valid, test) 

# UNCOMMENT TO GET PREDICTIONS

load_and_predict('jpg/image_08119.jpg', mapping[str(57)], 'model.h5')
