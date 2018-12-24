import os
import cv2
import numpy as np
import matplotlib

from keras_retinanet.models import convert_model

matplotlib.use("Agg")
import matplotlib.pyplot as plt

# import keras_retinanet
from keras_retinanet import models
from keras_retinanet.utils.colors import label_color
from keras_retinanet.utils.image import read_image_bgr, preprocess_image, resize_image
from keras_retinanet.utils.visualization import draw_box, draw_caption

image_dir="dataset/fire_1720/JPEGImages"
model_path = os.path.join('snapshots', 'retinanet_fire_2018-12-24_11:25.h5')

model=models.load_model(model_path, backbone_name='resnet50')
model=convert_model(model)
labels_to_names={0:'fire'}
num_file=20
test_file_path="dataset/fire_1720/ImageSets/Main/test.txt"
f=open(test_file_path,"r")
for line in f.readlines()[:num_file]:
    filename=line.strip()
    image=read_image_bgr(os.path.join(image_dir,filename+".jpg"))

    # copy to draw on
    draw = image.copy()
    draw = cv2.cvtColor(draw, cv2.COLOR_BGR2RGB)

    # preprocess image for network
    image = preprocess_image(image)
    image, scale = resize_image(image)

    boxes, scores, labels = model.predict_on_batch(np.expand_dims(image, axis=0))

    # correct for image scale
    boxes /= scale
    # visualize detections

    for box, score, label in zip(boxes[0], scores[0], labels[0]):
        # scores are sorted so we can break
        if score < 0.5:
            break

        color = label_color(label)

        b = box.astype(int)
        draw_box(draw, b, color = color)

        caption = "{} {:.3f}".format(labels_to_names[label], score)
        draw_caption(draw, b, caption)

    plt.figure(figsize = (20, 12))
    plt.imshow(draw)
    plt.axis('off')
    plt.savefig("figures/{}.jpg".format(filename))
f.close()