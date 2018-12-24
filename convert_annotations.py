import os
import xml.etree.ElementTree as ET

annotation_dir = "dataset/fire_1720/Annotations"
image_dir = "dataset/fire_1720/JPEGImages"
image_set_dir="dataset/fire_1720/ImageSets/Main"

for set in os.listdir(image_set_dir):
    counter=0
    setname=set.split(".")[0]
    new_annotation = open("{}_annotation.csv".format(setname), "w")
    set_file=open(os.path.join(image_set_dir,set))
    for line in set_file.readlines():
        counter+=1
        xml_filename=os.path.join(annotation_dir,"{}.xml".format(line))
        tree = ET.parse(os.path.join(annotation_dir,xml_filename))
        root = tree.getroot()
        filename = os.path.join(image_dir, root.find("filename").text)
        print(filename)
        objects = root.findall("object")
        for o in objects:
            c_name = o.find("name").text
            bnd_box = [i for i in o.find("bndbox").itertext() if i.strip() != '']
            new_annotation.write("{},{},{},{},{},{}\n".format(filename, *bnd_box, c_name))
    new_annotation.close()
    print("setname: {}, processed: {}".format(setname,counter))
