import numpy as np
import glob
import shutil
from os.path import join, basename, dirname, exists
import os
import json
import shlex,subprocess


def datapreprocess(dataset_path, output_path, rate=0.8):
    images = glob.glob("{}/*.jpg".format(dataset_path))
    class_names = set()
    dataset_dict = {}
    for image in images:
        image_filename = basename(image)
        class_name = "_".join(image_filename.split('_')[:-1])
        class_names.add(class_name)
        if class_name not in dataset_dict:
            dataset_dict[class_name] = [image]
        else:
            dataset_dict[class_name].append(image)
    if not exists(output_path):
        os.makedirs(output_path)
    data = sorted(list(class_names))
    class_to_num = {class_name: num for num, class_name in enumerate(data)}
    with open(join(output_path, 'label.json'), 'w') as f:
        json.dump(fp=f, obj=class_to_num)

    label_hander = {phase: open(
        join(output_path, '{}.txt'.format(phase)), 'w') for phase in ['train', 'val']}
    for key in dataset_dict:
        images = dataset_dict[key]
        num_train = int(len(images)*rate)
        split_image = {'train': images[:num_train], 'val': images[num_train:]}
        for phase in ['train', 'val']:
            output_path_tmp = join(output_path, phase)
            if not exists(output_path_tmp):
                os.makedirs(output_path_tmp)
            [shutil.copy(image, output_path_tmp)
             for image in split_image[phase]]
            [label_hander[phase].write('{}\n'.format(
                basename(image))) for image in split_image[phase]]
    [label_hander[key].close() for key in label_hander]
    result = {'train_image_label': join(
        output_path, 'train.txt'), 'val_image_label': join(output_path, 'val.txt'), 'label_to_num': join(output_path, 'label.json'), 'train_dataset': join(output_path, 'train'), 'val_dataset': join(output_path, 'val')}
    return result


def findfile(start, name):
    res = None
    for relpath, dirs, files in os.walk(start):
        if name in files:
            full_path = os.path.join(start, relpath, name)
            res = os.path.normpath(os.path.abspath(full_path))
    return res


def convert_dataset(output_res, dataset_store, shape=(225, 225)):
    caffe_home = os.path.expanduser('~/caffe')

    convert_tool = findfile(caffe_home, 'convert_imageset')
    assert convert_tool is not None, "Can't find convert_imageset"
    ouput_lmdb_train = join(dataset_store, 'train_lmdb')
    ouput_lmdb_val = join(dataset_store, 'val_lmdb')

    if exists(ouput_lmdb_train):
        shutil.rmtree(ouput_lmdb_train)
    if exists(ouput_lmdb_val):
        shutil.rmtree(ouput_lmdb_val)
    
    command_train = "{} --shuffle --resize_height={} --resize_width={}  {}/ {}  {}".format(
        convert_tool, shape[0], shape[1], output_res['train_dataset'],output_res['train_image_label'], ouput_lmdb_train)
    command_val = "{} --shuffle --resize_height={} --resize_width={}  {}/ {}  {}".format(
        convert_tool, shape[0], shape[1], output_res['train_dataset'],output_res['val_image_label'] ,ouput_lmdb_val)
    args_train = shlex.split(command_train)
    args_val = shlex.split(command_val)
    ferror = open('log.err','w')
    p = subprocess.Popen(args_train,stdout=ferror)
    p = subprocess.Popen(args_val,stdout=ferror)
    ferror.close()

    # ./.build_debug/tools/convert_imageset --shuffle --resize_height=256 --resize_width=256  /tmp/pet/train/ /tmp/pet/train.txt  /tmp/pet/img_train_lmdb



if __name__ == "__main__":
    dataset_path = "/home/liushuai/Datasets/pets/images"
    output_path = '/tmp/pet'
    output_path = datapreprocess(dataset_path, output_path)
    dataset_home = '/tmp'
    convert_dataset(output_path, dataset_home)
