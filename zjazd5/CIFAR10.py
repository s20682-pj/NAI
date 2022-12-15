""""
Authors: Zuzanna Ciborowska s20682 & Joanna Walkiewicz s20161
Python project for recognition of animals from the CIFAR collection using neural networks
System requirements:
- Python 3.10
- Tensorflow
- Numpy
- Keras
- Matplotlib
- Pandas
- Seaborn
"""

import numpy as np
import tensorflow as tf
from keras import datasets
from matplotlib import pyplot
from tensorflow.python import keras
import pandas
import seaborn

"""
Downloading and preparing the CIFAR10 datase
"""

cifar10 = tf.keras.datasets.cifar10

(train_images, train_labels), (test_images, test_labels) = datasets.cifar10.load_data()

train_images = train_images / 255.0
test_images = test_images / 255.0

"""
Verify the data
"""

class_names = ['airplane', 'automobile', 'bird', 'cat', 'deer','dog', 'frog', 'horse', 'ship', 'truck']

"""
Building a model
"""

model = keras.Sequential([
keras.layers.Conv2D(32, (3, 3), activation='relu', input_shape=(32, 32, 3)),
keras.layers.MaxPooling2D(2, 2),
keras.layers.Conv2D(64, (3, 3), activation='relu'),
keras.layers.MaxPooling2D((2, 2)),
keras.layers.Conv2D(64, (3, 3), activation='relu'),
keras.layers.Flatten(),
keras.layers.Dense(64, activation='relu'),
keras.layers.Dense(10),
    ])

"""
Compile and train the model
"""

model.compile(optimizer='adam',
              loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
              metrics=['accuracy'])

"""
Fit
"""

model.fit(train_images, train_labels, epochs=20, validation_data=(test_images, test_labels))

test_loss, test_acc = model.evaluate(test_images,  test_labels, verbose=2)
print('\nTest accuracy:', test_acc)


labels = '''airplane automobile bird cat deerdog frog horseship truck'''.split()

image_number = 0
pyplot.imshow(test_images[image_number])
pyplot.show()

n = np.array(test_images[image_number])
p = n.reshape(1, 32, 32, 3)
predicted_label = labels[model.predict(p).argmax()]
original_label = labels[test_labels[image_number][0]]
print("Original label is {} and predicted label is {}".format(
    original_label, predicted_label))

"""
prepare data frames for confusion matrix
"""
confusion_matrix = tf.math.confusion_matrix(labels=test_labels,
                                            predictions=np.argmax(model.predict(test_images), axis=1)
                                            ).numpy()

confusion_matrix_data_frames = pandas.DataFrame(
    data=np.around(confusion_matrix.astype('float') / confusion_matrix.sum(axis=1)[:, np.newaxis], decimals=2),
    index=class_names,
    columns=class_names
)
"""
Correlation Plot
"""

figure = pyplot.figure(figsize=(8, 8))
pyplot.tight_layout()

seaborn.heatmap(confusion_matrix_data_frames, annot=True)

"""
show heatmap
"""
pyplot.show()
