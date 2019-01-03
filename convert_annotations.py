import os
import xml.etree.ElementTree as ET
from os import mkdir
from os.path import isdir

dataset_name="hatman"

annotation_dir = "dataset/{}/Annotations".format(dataset_name)
image_dir = "dataset/{}/JPEGImages".format(dataset_name)
image_set_dir = "dataset/{}/ImageSets/Main".format(dataset_name)

labels={}
label_count=0
# create gound truth files for detection as well as converting annotations
if not isdir("retinanet_annotations/{}".format(dataset_name)):
    print("creating directory: " + "retinanet_annotations/{}".format(dataset_name))
    mkdir("retinanet_annotations/{}".format(dataset_name))

for set in os.listdir(image_set_dir):
    counter = 0
    setname = set.split(".")[0]
    new_annotation = open("retinanet_annotations/{}/{}_annotation.csv".format(dataset_name,setname), "w")
    set_file = open(os.path.join(image_set_dir, set))
    for line in set_file.readlines():
        counter += 1
        line = line.strip()
        xml_filename = os.path.join(annotation_dir, "{}.xml".format(line))
        gt_file=open("groundtruths/{}.txt".format(line),"w")
        tree = ET.parse(xml_filename)
        root = tree.getroot()
        filename = os.path.join("../..",image_dir, root.find("filename").text)
        objects = root.findall("object")
        for o in objects:
            c_name = o.find("name").text
            if c_name not in labels.keys():
                labels[c_name]=label_count
                label_count+=1
            bnd_box = [i for i in o.find("bndbox").itertext() if i.strip() != '']
            new_annotation.write("{},{},{},{},{},{}\n".format(filename, *bnd_box, c_name))
            gt_file.write("{} {} {} {} {}\n".format(c_name, *bnd_box))

        gt_file.close()
    new_annotation.close()
    print("setname: {}, processed: {}".format(setname, counter))

label_map=open("retinanet_annotations/{}/class_map.csv".format(dataset_name),"w")
for name, id in labels.items():
    label_map.write("{},{}\n".format(name,id))
print("labels\n",labels)