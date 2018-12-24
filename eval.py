import os
import cv2
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# import keras_retinanet
from keras_retinanet import models
from keras_retinanet.utils.colors import label_color
from keras_retinanet.utils.image import read_image_bgr, preprocess_image, resize_image
from keras_retinanet.utils.visualization import draw_box, draw_caption

model_path = os.path.join('snapshots', 'retinanet_fire_2018-12-24_11:25.h5')

model=models.load_model(model_path, backbone_name='resnet50')
labels_to_names={0:'fire'}
image=read_image_bgr('dataset/fire_1720/JPEGImages/rBOilFnymmWAKYFOAAQVF4R-xB8651.jpg')

draw = image.copy()
draw = cv2.cvtColor(draw, cv2.COLOR_BGR2RGB)

# preprocess image for network
image = preprocess_image(image)
image, scale = resize_image(image, min_side = 512, max_side = 512)
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

plt.figure(figsize = (15, 15))
plt.axis('off')
plt.imshow(draw)
plt.savefig("figures/test.jpg")