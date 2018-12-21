import matplotlib
matplotlib.use("Agg")

from keras_retinanet.preprocessing.csv_generator import CSVGenerator


# import keras
import keras

# import keras_retinanet
from keras_retinanet import models, losses

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


model = models.backbone('resnet50').retinanet(num_classes=1)
model.compile(
    loss={
        'regression'    : losses.smooth_l1(),
        'classification': losses.focal()
    },
    optimizer=keras.optimizers.adam(lr=1e-5, clipnorm=0.001)
)

weights="weights/resnet50_coco_best_v2.1.0.h5"
model.load_weights(weights, by_name=True, skip_mismatch=True)

data=CSVGenerator("annotation.csv","class_map.csv")
model.fit_generator(data, steps_per_epoch = 50, epochs = 20)
