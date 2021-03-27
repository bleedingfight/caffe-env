import numpy as np
import glob
import shutil
from os.path import join, basename, dirname, exists
import os
import json
import shlex
import subprocess
import caffe_pb2
import time
from google.protobuf import text_format
import argparse
import json


def datapreprocess(dataset_path, output_path, rate=0.8):
    class_names = sorted(os.listdir(dataset_path))
    with open(join(output_path, 'label.json'), 'w') as f:
        class_to_num = {class_name:num for num,class_name in class_names}
        json.dump(fp=f, obj=class_to_num)
    for class_name in class_names:
        images = os.listdir(dataset_path,class_name)
        train_num = int(len(images)*rate)
        train_image = images[:train_num]
        val_image = images[train_num:]
        [os.makedirs(join(output_path,phase),exist_ok=True) for phase in ['train','val']]
        [shutil.copy(join(dataset_path,class_name,image), join(output_path,'train'))
            for image in train_image]
        [shutil.copy(join(dataset_path,class_name,image), join(output_path,'val'))
            for image in val_image]
    
    for phase in ['train','val']:
        with open(join(output_path,phase),'w') as f:
            for image in os.listdir(output_path,'train'):
                f.write("{} {}".format())
    return result


def findfile(start, name):
    res = None
    for relpath, dirs, files in os.walk(start):
        if name in files:
            full_path = os.path.join(start, relpath, name)
            res = os.path.normpath(os.path.abspath(full_path))
    return res


def convert_dataset(dataset_path, dataset_store, shape=(227, 227)):
    output_res = datapreprocess(dataset_path, dataset_store)
    caffe_home = os.path.expanduser('~/caffe')
    if not os.path.exists(caffe_home):
        caffe_home = os.path.expanduser('~/caffe-env')

    convert_tool = findfile(caffe_home, 'convert_imageset')
    assert convert_tool is not None, "Can't find convert_imageset"
    if not exists(dataset_store):
        os.makedirs(dataset_store)
    for phase in ['train', 'val']:
        output_lmdb = join(dataset_store, '{}_lmdb'.format(phase))

        if exists(output_lmdb):
            shutil.rmtree(output_lmdb)

        command = "{} --shuffle --resize_height={} --resize_width={}  {}/ {}  {}".format(
            convert_tool, shape[0], shape[1], output_res['{}_dataset'.format(phase)], output_res['{}_image_label'.format(phase)], output_lmdb)
        output_res["{}_lmdb".format(phase)] = output_lmdb
        args = shlex.split(command)
        ferror = open('log.err', 'w')
        p_data = subprocess.Popen(args, stdout=ferror)
        compute_image_mean = findfile(caffe_home, 'compute_image_mean')
        if compute_image_mean is not None:
            command = "{} {} {}".format(compute_image_mean, output_lmdb+"/", join(
                dataset_store, 'mean_{}.binaryproto'.format(phase)))
            mean_args = shlex.split(command)
            p_data.wait()
            output_res['{}_mean'.format(phase)] = join(
                dataset_store, 'mean_{}.binaryproto'.format(phase))
            p = subprocess.Popen(mean_args)
    ferror.close()
    return output_res


def caffe_home():
    home_path = os.path.expanduser('~/')
    caffe_path = findfile(home_path, 'caffe')
    return caffe_path


def base_network(in_path, output_path, args):
    train_val = join(in_path, 'train_val.prototxt')
    network_module = caffe_pb2.NetParameter()
    solver_module = caffe_pb2.SolverParameter()
    with open(train_val, 'r') as f:
        text_format.Parse(f.read(), network_module)
    for layer in network_module.layer:
        if layer.type == 'Data':
            for phase_mesg in layer.include:
                if phase_mesg.phase == 0:
                    layer.transform_param.mean_file = args['train_mean']
                    layer.data_param.source = args['train_lmdb']
                    layer.data_param.batch_size = 16
                else:
                    layer.transform_param.mean_file = args['val_mean']
                    layer.data_param.source = args['val_lmdb']
        if layer.type == 'InnerProduct':
            if layer.inner_product_param.num_output == 1000:
                layer.inner_product_param.num_output = 37
    with open(join(output_path, 'train_val.pbtxt'), 'w') as f:
        f.write(text_format.MessageToString(network_module))

    backup_path = join(output_path, 'model')
    if not exists(backup_path):
        os.makedirs(backup_path)
    with open(join(in_path, 'solver.prototxt'), 'r') as f:
        text_format.Parse(f.read(), solver_module)
        solver_module.net = join(output_path, 'train_val.pbtxt')

        solver_module.snapshot_prefix = backup_path
    with open(join(output_path, 'solver.prototxt'),'w') as f:
        f.write(text_format.MessageToString(solver_module))
    # solver = join(in_path, 'solver.prototxt')


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='create parser parser data!')
    parser.add_argument('--output_path', '-o', default='/tmp/flowers', type=str)
    parser.add_argument('--dataset_path', '-d',
                        default='/home/liushuai/Datasets/flower_photos', type=str)
    parser.add_argument('--base_network', '-b',
                        default='/home/liushuai/caffe/models/bvlc_reference_caffenet', type=str)
    args = parser.parse_args()
    config = convert_dataset(args.dataset_path, args.output_path)
    with open(join(args.output_path, 'info.json'), 'w') as f:
        json.dump(obj=config, fp=f)

    with open(join(args.output_path, 'info.json'), 'r') as f:
        config = json.load(f)

    base_network(args.base_network, args.output_path, config)
