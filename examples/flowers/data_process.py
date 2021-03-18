import os
import shutil
import glob
import sys
from random import shuffle
import json
import caffe_pb2
from google.protobuf import text_format
import argparse

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


def parse_from_caffe(pbtxt, config=None):
    model = caffe_pb2.NetParameter()
    with open(pbtxt, 'r') as f:
        text_format.Parse(f.read(), model)
    dataset = [layer for layer in model.layer if layer.name == "data"]
    for data_layer in dataset:

        for elem in data_layer.include:
            if elem.phase == 0:
                data_layer.transform_param.mean_file = config['train_mean']
                data_layer.data_param.source = config['train_dataset']
            elif elem.phase == 1:
                data_layer.transform_param.mean_file = config['val_mean']
                data_layer.data_param.source = config['val_dataset']

        # print(data_layer.include.phase, dir(data_layer.include))
    for layer in model.layer:
        if layer.type == 'InnerProduct':
            layer.inner_product_param.num_output = config['class_nums']
    with open(config['output_train_val'], 'w') as f:
        f.write(text_format.MessageToString(model))
    return model


def parse_solver(solver, config):
    solver_message = caffe_pb2.SolverParameter()
    with open(solver, 'r') as f:
        text_format.Parse(f.read(), solver_message)
    solver_message.net = config['output_train_val']
    solver_message.snapshot_prefix = config['backup_dir']
    with open(config['output_solver'], 'w') as f:
        f.write(text_format.MessageToString(solver_message))
    return solver_message


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="create caffe dataset")
    parser.add_argument('--dataset_path', '-d', help="Datasets path")
    parser.add_argument('--output_path', '-o', help="Processed dataset path")
    parser.add_argument('--train_val', '-t',
                        help="train and val prototxt file")
    parser.add_argument('--solver', '-s', help="caffe solver")
    parser.add_argument('--debug', '-de', default=False, type=bool)
    parser.add_argument('--class_nums','-c',help='classes num',default=5)
    parser.add_argument('--output_train_val','-otv',help='generator train_val.pbtxt',default='/tmp/flower_dataset')
    parser.add_argument('--backup_dir','-bd',help='model saved path',default='/tmp/flower_dataset/')
    parser.add_argument('--output_solver','-os',help='solver.pbtxt saved path',default='/tmp/flower_dataset/solver.pbtxt')
    args = parser.parse_args()
    process_data(args.dataset_path, args.output_path)
    config = {'train_mean': "mean_train.bin", "val_mean": "mean_val.bin", "class_nums": 5,
              'train_dataset': "/tmp/train_val_lmdb", "val_dataset": "/tmp/train_val_lmdb", "output_train_val": "/tmp/train_val.prototxt", "backup_dir": '/tmp/backup', 'output_solver': "/tmp/solver.pbtxt"}
    if args.debug:
        parse_from_caffe(args.train_val, config)
        parse_solver(args.solver, config)
        print("Generator train_val:{} parse:{}".format(
            config['output_train_val'], config['output_solver']))
