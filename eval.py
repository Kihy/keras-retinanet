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

model_path = os.path.join('snapshots', 'retinanet_fire_2018-12-24_11:25.h5')

model=models.load_model(model_path, backbone_name='resnet50')
model=convert_model(model)

labels_to_names={0:'fire'}
image=read_image_bgr('dataset/fire_1720/JPEGImages/rBOilFnymmWAKYFOAAQVF4R-xB8651.jpg')

draw = image.copy()
draw = cv2.cvtColor(draw, cv2.COLOR_BGR2RGB)

# preprocess image for network
image = preprocess_image(image)
image, scale = resize_image(image, min_side = 512, max_side = 512)
boxes, scores, labels = model.predict_on_batch(np.expand_dims(image, axis=0))
n_classes=1
# correct for image scale
boxes /= scale
# visualize detections
current_axis = plt.gca()
colors = plt.cm.hsv(np.linspace(0, 1, n_classes + 1)).tolist()
for box, score, label in zip(boxes[0], scores[0], labels[0]):
    # scores are sorted so we can break
    if score < 0.5:
        break

    color = colors[label]
    print(box)
    xmin = box[0]
    ymin = box[1]
    xmax = box[2]
    ymax = box[3]

    caption = "{} {:.3f}".format(labels_to_names[label], score)
    current_axis.add_patch(
        plt.Rectangle((xmin, ymin), xmax - xmin, ymax - ymin, color = color, fill = False, linewidth = 2))
    current_axis.text(xmin, ymin, label, size = 'x-large', color = 'white', bbox = {'facecolor': color, 'alpha': 1.0})

plt.figure(figsize = (15, 15))
plt.axis('off')
plt.imshow(draw)
plt.savefig("figures/test.jpg")