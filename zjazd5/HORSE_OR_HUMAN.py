""""
Authors: Zuzanna Ciborowska s20682 & Joanna Walkiewicz s20161
Python project to recognize whether there is a human or a horse in the picture using neural networks
System requirements:
- Python 3.10
- Tensorflow
- Keras
"""

from tensorflow.keras.optimizers import RMSprop
import tensorflow_datasets as tfds
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import os

"""
Download zip with dataset and unzip files
"""
dl_manager = tfds.download.DownloadManager(download_dir='./tmp')

data_dirs = dl_manager.download_and_extract({
   'train': 'https://storage.googleapis.com/laurencemoroney-blog.appspot.com/horse-or-human.zip',
   'test': 'https://storage.googleapis.com/laurencemoroney-blog.appspot.com/validation-horse-or-human.zip',
})

train_horse_dir = './tmp/extracted/' + data_dirs['train'].name + '/horses'
train_human_dir = './tmp/extracted/' + data_dirs['train'].name + '/humans'
valid_horse_dir = './tmp/extracted/' + data_dirs['test'].name + '/horses'
valid_human_dir = './tmp/extracted/' + data_dirs['test'].name + '/humans'

"""
Print number of horse and human images for training, number horse and human validation
"""
print('The total number of horse images for training : ', len(os.listdir(train_horse_dir)))
print('The total number of human images for training : ', len(os.listdir(train_human_dir)))
print('The total number of horse images for validation : ', len(os.listdir(valid_horse_dir)))
print('The total number of human images for validation : ', len(os.listdir(valid_human_dir)))

"""
Building a model
"""
model = tf.keras.Sequential([# First Convolution layer
                             tf.keras.layers.Conv2D(16, (3, 3), activation=tf.nn.relu, input_shape=(300, 300, 3)),
                             tf.keras.layers.MaxPool2D(2, 2),
                             # Second Convolution layer
                             tf.keras.layers.Conv2D(32, (3, 3), activation=tf.nn.relu),
                             tf.keras.layers.MaxPool2D(2, 2),
                             # Third Convolution layer
                             tf.keras.layers.Conv2D(64, (3, 3), activation=tf.nn.relu),
                             tf.keras.layers.MaxPool2D(2, 2),
                             # Fourth Convolution layer
                             tf.keras.layers.Conv2D(64, (3, 3), activation=tf.nn.relu),
                             tf.keras.layers.MaxPool2D(2, 2),
                             # Fifth Convolution layer
                             tf.keras.layers.Conv2D(64, (3, 3), activation=tf.nn.relu),
                             tf.keras.layers.MaxPool2D(2, 2),
                             # Flatten and feed into DNN
                             tf.keras.layers.Flatten(),
                             tf.keras.layers.Dense(512, activation=tf.nn.relu),
                             tf.keras.layers.Dense(1, activation=tf.nn.sigmoid)])

"""
Define the optimizers and the loss function which it will be using in the model
"""
model.compile(optimizer = RMSprop(learning_rate=0.001), loss='binary_crossentropy', metrics=['accuracy'])

"""
Model summary
"""
model.summary()

"""
Performing data preprocessing
"""
train_datagen = ImageDataGenerator(1/255)
validation_datagen = ImageDataGenerator(1/255)
train_generator = train_datagen.flow_from_directory('./tmp/extracted/' + data_dirs['train'].name,
                                                    target_size=(300, 300),
                                                    batch_size=128,
                                                    class_mode='binary')

validation_generator = validation_datagen.flow_from_directory('./tmp/extracted/' + data_dirs['test'].name,
                                                              target_size=(300, 300),
                                                              batch_size=32,
                                                              class_mode='binary')
"""
Defining callbacks witch will help in the early stopping of training if model has reached the desired accuarcy
"""
class myCallback(tf.keras.callbacks.Callback):
  def on_epoch_end(self, epoch, logs={}):
    if(logs.get('loss')<0.10):
      print("\nReached 90% accuracy so cancelling training!")
      self.model.stop_training = True

callbacks = myCallback()

"""
Training model
"""
history = model.fit(
    train_generator,
    steps_per_epoch=8,
    epochs=15,
    verbose=1,
    validation_data=validation_generator,
    validation_steps=8,
    callbacks=[callbacks]
)