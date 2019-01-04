from os import listdir, mkdir, rename
from os.path import join, splitext, isdir, isfile
import random
from configparser import ConfigParser, ExtendedInterpolation


def main():
    """
    converts data from the internal server to pascal voc dataset
    format. The dataset downloaded from server is stored with images and annotations in the same folder
    this script organises them into three folders, JPEGImages, Annotations and ImageSets
    :return:
    """
    parser = ConfigParser(interpolation = ExtendedInterpolation())
    parser.read("dataset_config.ini")
    params = parser["imageset"]

    # Path to dataset folder
    data_folder = params["data_folder"]

    # Split Percentage
    train_prob = float(params["train"])
    val_prob = float(params["val"])
    test_prob = float(params["test"])  # there is a seperate imageset for testing thus it is set to 0

    # sanity check, allow for partial splits
    assert train_prob + val_prob + test_prob <= 1

    # Seeding
    random.seed(1)

    folder_names = ["JPEGImages", "ImageSets", "Annotations", "ImageSets/Main"]
    # make folders
    for name in folder_names:
        folder = join(data_folder, name)
        if not isdir(folder):
            print("creating directory: " + folder)
            mkdir(folder)

    # move file to right place
    for filename in listdir(data_folder):
        if isfile(join(data_folder, filename)):
            if filename.endswith(".jpg"):
                rename(join(data_folder, filename), join(data_folder, "JPEGImages", filename))

            elif filename.endswith(".xml"):
                rename(join(data_folder, filename), join(data_folder, "Annotations", filename))

    # create imageset
    # Path to all images files
    image_path = join(data_folder, "JPEGImages")
    # Path for the output of the image sets
    imageset_path = join(data_folder, "ImageSets/Main")

    filenames = [splitext(f)[0] for f in listdir(image_path)]

    # Calculate number of files for each
    num_file = len(filenames)
    train_cutoff = int(train_prob * num_file)
    val_cutoff = int((train_prob + val_prob) * num_file)
    random.shuffle(filenames)

    # open files
    train_file = open(join(imageset_path, "train.txt"), "w")
    test_file = open(join(imageset_path, "test.txt"), "w")
    val_file = open(join(imageset_path, "val.txt"), "w")

    train_list = filenames[:train_cutoff]
    val_list = filenames[train_cutoff:val_cutoff]
    test_list = filenames[val_cutoff:]

    train_file.write("\n".join(train_list))
    test_file.write("\n".join(test_list))
    val_file.write("\n".join(val_list))

    train_file.close()
    test_file.close()
    val_file.close()

    print("#train: {}, #val: {}, #test:{}".format(len(train_list), len(val_list), len(test_list)))


if __name__ == "__main__":
    main()
