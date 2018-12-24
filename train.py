from datetime import datetime
import os

import matplotlib
from keras.callbacks import ModelCheckpoint, CSVLogger, TerminateOnNaN, TensorBoard

matplotlib.use("Agg")

from keras_retinanet.preprocessing.csv_generator import CSVGenerator


# import keras
import keras
from keras.backend import tensorflow_backend as ktf
# import keras_retinanet
from keras_retinanet import models, losses

# set tf backend to allow memory to grow, instead of claiming everything
import tensorflow as tf

def reize( size):
    return lambda x: ktf.reshape(x, size)

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

data = CSVGenerator("annotation.csv","class_map.csv", batch_size = 16, image_min_side = 512, image_max_side = 512, preprocess_image = reize((512,512,3)))

# TODO: Set the file path under which you want to save the model.
current_time = datetime.now().strftime('%Y-%m-%d %H:%M').split(" ")
model_checkpoint = ModelCheckpoint(
    filepath = os.path.join("snapshots",
                            'retinanet_fire_{}_{}.h5'.format(current_time[0], current_time[1])),
    monitor = 'val_loss',
    verbose = 1,
    save_best_only = True,
    save_weights_only = False,
    mode = 'auto',
    period = 10)
# model_checkpoint.best =

csv_logger = CSVLogger(
    filename = os.path.join("csv_logs", 'retinanet_fire_{}_{}.csv'.format(current_time[0], current_time[1])),
    separator = ',',
    append = True)

# learning_rate_scheduler = LearningRateScheduler(schedule=lr_schedule, verbose=1)

terminate_on_nan = TerminateOnNaN()

tensorboard = TensorBoard(
    log_dir = os.path.join("tensorboard", 'retinanet', current_time[0], current_time[1]),
    write_images = True, write_graph = True)

callbacks = [model_checkpoint,
             csv_logger,
             #            learning_rate_scheduler,
             terminate_on_nan,
             tensorboard]
# Fit model

# If you're resuming a previous training, set `initial_epoch` and `final_epoch` accordingly.
initial_epoch = 0
final_epoch = 20
steps_per_epoch = 86

history = model.fit_generator(generator = data,
                              steps_per_epoch = steps_per_epoch,
                              epochs = final_epoch,
                              callbacks = callbacks,
                              # validation_data = val_generator,
                              # validation_steps = ceil(val_dataset_size / batch_size),
                              initial_epoch = initial_epoch)


