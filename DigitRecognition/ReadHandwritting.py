import tensorflow as tf
import PIL.Image
import numpy as np
import os

# Load Model:
model = tf.keras.models.Sequential([
  tf.keras.layers.Flatten(name='f'),
  tf.keras.layers.Dense(512, activation=tf.nn.relu, name='d1',  weights=np.load("weights1.npy")),
  tf.keras.layers.Dropout(0.2, name='dr'),
  tf.keras.layers.Dense(10, activation=tf.nn.softmax,name='d2', weights=np.load("weights3.npy"))
])
model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

os.system('bash -c '+ 'pinta num.png &');



import sys
import time
from watchdog.observers import Observer
from watchdog.events import LoggingEventHandler

class Event(LoggingEventHandler):
    def dispatch(self, event):
        self.predict()

    def predict(self):
        # make a prediction
        img = self.loadImg()
        prediction = model.predict_classes(img)
        # show the inputs and predicted outputs
        print("\n\n\n\n--------------\nPAL9K says that's a %s!\n--------------" % prediction[0])

    def loadImg(self):
        img = PIL.Image.open("num.png")
        img = img.rotate(-90).transpose(PIL.Image.FLIP_LEFT_RIGHT)
        pixels = img.load()  # this is not a list, nor is it list()'able
        width, height = img.size

        all_pixels = []

        for x in range(width):
            for y in range(height):
                cpixel = pixels[x, y]
                r = cpixel[0]
                g = cpixel[1]
                b = cpixel[2]
                grayPix = ((r + g + b) / 3)
                all_pixels.append(grayPix)

        img = all_pixels
        img = np.expand_dims(img, axis=0)

        # test_data = np.array([0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,10,17,17,17,17,81,180,180,35,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,139,253,253,253,253,253,253,253,48,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,60,228,253,253,253,253,253,253,253,207,197,46,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,213,253,253,253,253,253,253,253,253,253,253,223,52,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,66,231,253,253,253,108,40,40,115,244,253,253,134,3,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,63,114,114,114,37,0,0,0,205,253,253,253,15,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,57,253,253,253,15,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,42,253,253,253,15,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,95,253,253,253,15,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,205,253,253,253,15,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,61,99,96,0,0,45,224,253,253,195,10,0,0,0,0,0,0,0,0,0,0,0,11,25,105,83,189,189,228,253,251,189,189,218,253,253,210,27,0,0,0,0,0,0,0,0,0,0,42,116,173,253,253,253,253,253,253,253,253,253,253,253,253,253,221,116,7,0,0,0,0,0,0,0,0,0,118,253,253,253,253,245,212,222,253,253,253,253,253,253,253,253,253,253,160,15,0,0,0,0,0,0,0,0,254,253,253,253,189,99,0,32,202,253,253,253,240,122,122,190,253,253,253,174,0,0,0,0,0,0,0,0,255,253,253,253,238,222,222,222,241,253,253,230,70,0,0,17,175,229,253,253,0,0,0,0,0,0,0,0,158,253,253,253,253,253,253,253,253,205,106,65,0,0,0,0,0,62,244,157,0,0,0,0,0,0,0,0,6,26,179,179,179,179,179,30,15,10,0,0,0,0,0,0,0,0,14,6,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0])
        # img_array = np.reshape(test_data, (28,28))
        # save_img = PIL.Image.fromarray(np.uint8(np.array(img_array).reshape(28,28)) , 'L')
        # save_img.save("saved.png")

        img = np.expand_dims(img, axis=0)
        return img


if __name__ == "__main__":
    path = sys.argv[1] if len(sys.argv) > 1 else '.'
    event_handler = Event()
    observer = Observer()
    observer.schedule(event_handler, path, recursive=True)
    observer.start()
    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        observer.stop()
    observer.join()



