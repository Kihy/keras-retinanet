import os
from configparser import ConfigParser, ExtendedInterpolation
from datetime import datetime
from math import ceil

import matplotlib
from keras.callbacks import ModelCheckpoint, CSVLogger, TerminateOnNaN, TensorBoard
from keras_retinanet.utils.image import preprocess_image

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
    return tf.Session(config = config)


def get_preprocessing(flag = False):
    if flag:
        return lambda x: preprocess_image(x, mode = 'tf')
    else:
        return lambda x: x


def main():
    parser = ConfigParser(interpolation = ExtendedInterpolation())
    parser.read("config.ini")
    params = parser["train"]
    backbone = params["backbone"]

    # use this environment flag to change which GPU to use
    # os.environ["CUDA_VISIBLE_DEVICES"] = "1"

    # set the modified tf session as backend in keras
    keras.backend.tensorflow_backend.set_session(get_session())

    model = models.backbone(backbone).retinanet(num_classes = 2)
    model.compile(
        loss = {
            'regression': losses.smooth_l1(),
            'classification': losses.focal()
        },
        optimizer = keras.optimizers.adam(lr = 1e-5, clipnorm = 0.001),
        metrics = ['accuracy']
    )

    weights = params["weights_path"]
    model.load_weights(weights, by_name = True, skip_mismatch = True)
    batch_size = int(params["batchsize"])

    preprocesss = get_preprocessing(bool(params["preprocess"]))

    path = "retinanet_annotations/{}".format(params["dataset"])
    data = CSVGenerator(os.path.join(path, "train_annotation.csv"), os.path.join(path, "class_map.csv"),
                        batch_size = batch_size, image_min_side = 512,
                        image_max_side = 512, preprocess_image = preprocesss)
    val_data = CSVGenerator(os.path.join(path, "val_annotation.csv"), os.path.join(path, "class_map.csv"),
                            batch_size = batch_size, image_min_side = 512,
                            image_max_side = 512, preprocess_image = preprocesss)

    val_dataset_size = val_data.size()

    # TODO: Set the file path under which you want to save the model.
    current_time = datetime.now().strftime('%Y-%m-%d %H:%M').split(" ")
    model_checkpoint = ModelCheckpoint(
        filepath = os.path.join(params["check_point_path"],
                                'retinanet_{}_{}_{}_{}_{}.h5'.format(params["dataset"], backbone, params["preprocess"],
                                                                     current_time[0], current_time[1])),
        monitor = 'val_loss',
        verbose = 1,
        save_best_only = True,
        save_weights_only = False,
        mode = 'auto',
        period = int(params["save_period"]))
    # model_checkpoint.best =

    csv_logger = CSVLogger(
        filename = os.path.join(params["csv_path"],
                                'retinanet_{}_{}_{}_{}_{}.csv'.format(params["dataset"], backbone, params["preprocess"],
                                                                      current_time[0], current_time[1])),
        separator = ',',
        append = True)

    # learning_rate_scheduler = LearningRateScheduler(schedule=lr_schedule, verbose=1)

    terminate_on_nan = TerminateOnNaN()

    tensorboard = TensorBoard(
        log_dir = os.path.join(params["tensorboard_path"], 'retinanet', current_time[0], current_time[1]),
        write_images = True, write_graph = True)

    callbacks = [model_checkpoint,
                 csv_logger,
                 #            learning_rate_scheduler,
                 terminate_on_nan,
                 tensorboard,
                 ]
    # Fit model

    # If you're resuming a previous training, set `initial_epoch` and `final_epoch` accordingly.
    initial_epoch = 0
    final_epoch = int(params["epochs"])
    steps_per_epoch = int(params["steps_per_epoch"])

    model.fit_generator(generator = data,
                        steps_per_epoch = steps_per_epoch,
                        epochs = final_epoch,
                        callbacks = callbacks,
                        validation_data = val_data,
                        validation_steps = ceil(val_dataset_size / batch_size),
                        initial_epoch = initial_epoch,
                        )


if __name__ == '__main__':
    main()
