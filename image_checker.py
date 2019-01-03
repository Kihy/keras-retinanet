import os
from PIL import Image
import numpy as np

image_dir="dataset/hatman/JPEGImages"
count=0
for i in os.listdir(image_dir):
    count+=1
    print(count)
    try:
        im=np.asarray(Image.open(os.path.join(image_dir,i)).convert('RGB'))
    except Exception as e:
        print(i)
        print(e)
