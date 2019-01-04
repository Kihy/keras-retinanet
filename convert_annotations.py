import os
import xml.etree.ElementTree as ET
from os import mkdir
from os.path import isdir
from configparser import ConfigParser, ExtendedInterpolation


def main():
    """
    converts annotations into the format used for csv paraser in retinanet.
    the annotations should already been in pascal voc format, otherwise turn it into
    pascal voc format by running imageset_generator.py
    also creates ground truth files for calculating metrics
    :return:
    """

    parser = ConfigParser(interpolation = ExtendedInterpolation())
    parser.read("dataset_config.ini")
    params = parser["convert_anno"]

    dataset_name = params["dataset_name"]
    dataset_dir = os.path.join("dataset", dataset_name)

    annotation_dir = os.path.join(dataset_dir, "Annotations")
    image_dir = os.path.join(dataset_dir, "JPEGImages")
    image_set_dir = os.path.join(dataset_dir, "ImageSets", "Main")

    labels = {}
    label_count = 0

    output_dir = os.path.join("retinanet_annotations", dataset_name)
    if not isdir(output_dir):
        print("creating directory: " + output_dir)
        mkdir(output_dir)

    for set in os.listdir(image_set_dir):
        counter = 0
        setname = set.split(".")[0]
        new_annotation = open(
            os.path.join(os.path.join("retinanet_annotations", dataset_name, "{}_annotation".format(setname))), "w")
        set_file = open(os.path.join(image_set_dir, set))
        for line in set_file.readlines():
            counter += 1
            line = line.strip()
            xml_filename = os.path.join(annotation_dir, "{}.xml".format(line))
            gt_file = open("groundtruths/{}.txt".format(line), "w")
            tree = ET.parse(xml_filename)
            root = tree.getroot()
            filename = os.path.join("../..", image_dir, root.find("filename").text)
            objects = root.findall("object")
            for o in objects:
                c_name = o.find("name").text
                if c_name not in labels.keys():
                    labels[c_name] = label_count
                    label_count += 1
                bnd_box = [i for i in o.find("bndbox").itertext() if i.strip() != '']
                new_annotation.write("{},{},{},{},{},{}\n".format(filename, *bnd_box, c_name))
                gt_file.write("{} {} {} {} {}\n".format(c_name, *bnd_box))

            gt_file.close()
        new_annotation.close()
        print("set name: {}, processed: {}".format(setname, counter))

    label_map = open(os.path.join(output_dir,"class_map.csv"), "w")
    for name, label_id in labels.items():
        label_map.write("{},{}\n".format(name, label_id))
    print("labels\n", labels)


if __name__ == '__main__':
    main()
