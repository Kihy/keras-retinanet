import os
import xml.etree.ElementTree as ET

annotation_dir = "dataset/fire_1720/Annotations"
image_dir = "dataset/fire_1720/JPEGImages"

new_annotation = open("annotation.csv", "w")
for file in os.listdir(annotation_dir):
    if file.endswith(".xml"):
        tree = ET.parse(file)
        root = tree.getroot()
        filename = os.path.join(image_dir, root.find("filename").text)
        print(filename)
        objects = root.findall("object")
        for o in objects:
            c_name = o.find("name").text
            bnd_box = [i for i in o.find("bndbox").itertext() if i.strip() != '']
            print(c_name)
            print(bnd_box)
            new_annotation.write("{},{},{},{},{},{}\n".format(filename, *bnd_box, c_name))

new_annotation.close()
