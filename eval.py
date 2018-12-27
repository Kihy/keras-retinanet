import os
from configparser import ConfigParser, ExtendedInterpolation

import cv2
import numpy as np
import matplotlib

from keras_retinanet.models import convert_model
from train import get_preprocessing

matplotlib.use("Agg")
import matplotlib.pyplot as plt

# import keras_retinanet
from keras_retinanet import models
from keras_retinanet.utils.colors import label_color
from keras_retinanet.utils.image import read_image_bgr, preprocess_image, resize_image
from keras_retinanet.utils.visualization import draw_box, draw_caption

def create_ground_truths(path, image_names):
    annotations = open(path, "r")
    anno_dict = {}
    for i in annotations.readlines():
        if i.rstrip() not in image_names:
            break
        name, b1, b2, b3, b4, cls = i.rstrip().split()
        if name not in anno_dict.keys():
            anno_dict[name] = []
        anno_dict[name].append([b1, b2, b3, b4])
    return anno_dict



def main():

    parser = ConfigParser(interpolation = ExtendedInterpolation())
    parser.read("config.ini")
    params = parser["eval"]

    image_dir = params["image_dir"]
    model_path = params["model_path"]
    print(params["backbone"])
    model = models.load_model(model_path, backbone_name = params["backbone"])
    model = convert_model(model)
    labels_to_names = {0: 'fire'}
    n_classes = len(labels_to_names)
    num_file = int(params["num_file"])
    test_file_path = params["image_set_path"]
    f = open(test_file_path, "r")
    image_names = f.readlines()[:num_file]
    f.close()
    annotations_dict = load_annotations(params["annotation_path"], image_names)

    for line in image_names:
        filename = line.strip()
        image = matplotlib.image.imread(os.path.join(image_dir, filename + ".jpg"))
        gt_box = annotations_dict[filename]
        # # copy to draw on
        # draw = image.copy()
        # draw = cv2.cvtColor(draw, cv2.COLOR_BGR2RGB)
        plt.imshow(image)
        print(filename)

        # preprocess image for network
        prep_function = get_preprocessing()

        image = prep_function(image)
        image, scale = resize_image(image, min_side = 512, max_side = 512)

        boxes, scores, labels = model.predict_on_batch(np.expand_dims(image, axis = 0))

        # correct for image scale
        boxes /= scale
        # visualize detections
        ax = plt.gca()
        for box, score, label in zip(boxes[0], scores[0], labels[0]):
            # scores are sorted so we can break

            if score < 0.5:
                print("ignored: ", box, score)
                break

            colors = plt.cm.hsv(np.linspace(0, 1, n_classes + 1)).tolist()

            b = box.astype(int)
            print(b)
            # Create a Rectangle patch
            rect = matplotlib.patches.Rectangle((b[0], b[1]), (b[2] - b[0]), (b[3] - b[1]), linewidth = 1,
                                                color = colors[label], fill = False)

            # Add the patch to the Axes
            ax.add_patch(rect)
            # draw_box(image, b, color = color)

            caption = "{} {:.3f}".format(labels_to_names[label], score)
            plt.text(b[0] + 5, b[1] - 12, caption, bbox = dict(boxstyle = "square", color = colors[label]))

        # plt.axis('off')
        plt.savefig("figures/{}.jpg".format(filename))
        plt.clf()

