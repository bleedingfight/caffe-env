import os
import shutil
import glob
import sys
from random import shuffle


def write_labels(label_files, labels, label_to_index):
    with open(label_files, 'a') as f:
        for label in labels:
            filename = os.path.basename(os.path.dirname(label))
            info = "{} {}\n".format(label, label_to_index[filename])
            f.write(info)


def copy_file_to(filenames, dest):
    for filename in filenames:
        dest_filename = os.path.join(dest, os.path.basename(filename))
        shutil.copyfile(filename, dest_filename)


def process_data(dataset_path, export_path, fmt='jpg', split=0.8):
    classes = {}
    if not os.path.exists(export_path):
        os.makedirs(export_path)
    [classes.update({element[1]: element[0]})
     for element in enumerate(sorted(os.listdir(dataset_path)))]
    for class_name in classes:
        image_lists = glob.glob(
            dataset_path + "/{}/*.{}".format(class_name, fmt))
        shuffle(image_lists)
        train_nums = int(len(image_lists) * split)
        train_images_path = image_lists[:train_nums]
        test_images_path = image_lists[train_nums:]
        export_class_path = [os.path.join(export_path, mode, class_name) for mode in [
            'train', 'test']]
        for class_export_path in export_class_path:
            if not os.path.exists(class_export_path):
                os.makedirs(class_export_path)
            current_class = os.path.basename(class_export_path)
            if class_export_path.count('train'):
                copy_file_to(train_images_path, class_export_path)
            else:
                copy_file_to(test_images_path, class_export_path)

        train_labels_file = os.path.join(export_path, 'train.txt')
        test_labels_file = os.path.join(export_path, 'test.txt')
        write_labels(train_labels_file, train_images_path, classes)
        write_labels(test_labels_file, test_images_path, classes)


if __name__ == "__main__":
    dataset_path = '/home/liushuai/flower_photos/flower'
    export_path = '/tmp/flower_photos/export'
    process_data(dataset_path, export_path)
