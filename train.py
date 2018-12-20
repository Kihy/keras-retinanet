import matplotlib

from keras_retinanet.preprocessing.csv_generator import CSVGenerator

matplotlib.use("Agg")
# import keras
import keras

# import keras_retinanet
import keras_retinanet


# import miscellaneous modules
import matplotlib.pyplot as plt
import cv2
import os
import numpy as np
import time

# set tf backend to allow memory to grow, instead of claiming everything
import tensorflow as tf

def get_session():
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    return tf.Session(config=config)

# use this environment flag to change which GPU to use
#os.environ["CUDA_VISIBLE_DEVICES"] = "1"

# set the modified tf session as backend in keras
keras.backend.tensorflow_backend.set_session(get_session())


model = keras_retinanet.models.backbone('resnet50').retinanet(num_classes=1)
model.compile(
    loss={
        'regression'    : keras_retinanet.losses.smooth_l1(),
        'classification': keras_retinanet.losses.focal()
    },
    optimizer=keras.optimizers.adam(lr=1e-5, clipnorm=0.001)
)

data=CSVGenerator("annotation.csv","class_map.csv")
model.fit_generator(data, steps_per_epoch = 50, epochs = 20)