import tensorflow as tf
from tensorflow.keras.models import load_model
from PIL import Image
import h5py
import inspect
import numpy as np

# Launch a session.
sess = tf.Session()


mnist = tf.keras.datasets.mnist

(x_train, y_train),(x_test, y_test) = mnist.load_data()
x_train, x_test = x_train / 255.0, x_test / 255.0

model = tf.keras.models.Sequential([
  tf.keras.layers.Flatten(name='layer1'),
  tf.keras.layers.Dense(512, activation=tf.nn.relu, name='layer2'),
  tf.keras.layers.Dropout(0.2, name='layer3'),
  tf.keras.layers.Dense(10, activation=tf.nn.softmax, name='layer4')
])
model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])



model.fit(x_train, y_train, epochs=5)
model.evaluate(x_test, y_test)

for i in range(len(model.layers)):
    filename = "weights"+str(i)+".npy"
    np.save(filename, model.layers[i].get_weights())

model.save_weights("results/mnist.h5")



print("SAVING-----------------------------------")
model.save('saved_model.h5')  # creates a HDF5 file 'my_model.h5'
del model  # deletes the existing model


#
# # serialize model to JSON
# model_json = model.to_json()
# with open("model.json", "w") as json_file:
#     json_file.write(model_json)
# # serialize weights to HDF5
# model.save_weights("model.h5")
# print("Saved model to disk")
