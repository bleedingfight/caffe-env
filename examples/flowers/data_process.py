import os
import shutil
import glob
import sys
from random import shuffle
import json

# 将数据的图片位置和label数值写入文件


def write_labels(label_files, labels, label_to_index):
    with open(label_files, 'a') as f:
        for label in labels:
            filename = os.path.basename(os.path.dirname(label))
            info = "{} {}\n".format(label, label_to_index[filename])
            f.write(info)

# 将处理好的训练和测试数据写入指定目录


def copy_file_to(filenames, dest):
    for filename in filenames:
        dest_filename = os.path.join(dest, os.path.basename(filename))
        shutil.copyfile(filename, dest_filename)

# 对数据集中的数据shuffle、split存放


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


def convert_to_lmdb(config_info):
    export_path = config_info['export_data_path']
    dataset_path = config_info['dataset_path']
    if os.path.exists(export_path):
        shutil.rmtree(export_path, ignore_errors=True)
    else:
        os.makedirs(export_path)
    sys.stdout.write(
        "Processing data to :\033[31m{}\033[0m\n".format(export_path))

    process_data(dataset_path, export_path)
    resized = config_info[resized_shape]
    if resized is not None:
        sys.stdout.write(
            "Dataset will be resized as:\033\31m[{}]\033[0m\n".format(resized))
        convert_command = "glog_logtostderr=1 {} --resize_height {:d} --resize_width {:d} --shuffle {} {} {}".format(
            config_info['convert_program'], resized[0], resized[1], config_info['dateset_path_prefix'], config_info['train_txt_path'], config_info['train_lmdb_path'])
        pass


if __name__ == "__main__":
    dataset_path = '/home/liushuai/flower_photos/flower'
    export_path = '/tmp/flower_photos/export'
    process_data(dataset_path, export_path)
