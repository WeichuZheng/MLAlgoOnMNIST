import struct
import numpy as np
import os
import cv2


def decode_idx3_ubyte(idx3_ubyte_file):
    with open(idx3_ubyte_file, 'rb') as f:
        print('解析文件：', idx3_ubyte_file)
        fb_data = f.read()

    offset = 0
    fmt_header = '>iiii'  # 以大端法读取4个 unsigned int32
    magic_number, num_images, num_rows, num_cols = struct.unpack_from(fmt_header, fb_data, offset)
    print('魔数：{}，图片数：{}'.format(magic_number, num_images))
    offset += struct.calcsize(fmt_header)
    fmt_image = '>' + str(num_rows * num_cols) + 'B'

    images = np.empty((num_images, num_rows, num_cols))
    for i in range(num_images):
        im = struct.unpack_from(fmt_image, fb_data, offset)
        images[i] = np.array(im).reshape((num_rows, num_cols))
        offset += struct.calcsize(fmt_image)
    return images


def decode_idx1_ubyte(idx1_ubyte_file):
    with open(idx1_ubyte_file, 'rb') as f:
        print('解析文件：', idx1_ubyte_file)
        fb_data = f.read()

    offset = 0
    fmt_header = '>ii'  # 以大端法读取两个 unsigned int32
    magic_number, label_num = struct.unpack_from(fmt_header, fb_data, offset)
    print('魔数：{}，标签数：{}'.format(magic_number, label_num))
    offset += struct.calcsize(fmt_header)
    labels = []

    fmt_label = '>B'  # 每次读取一个 byte
    for i in range(label_num):
        labels.append(struct.unpack_from(fmt_label, fb_data, offset)[0])
        offset += struct.calcsize(fmt_label)
    return labels


def check_folder(folder):
    """检查文件文件夹是否存在，不存在则创建"""
    if not os.path.exists(folder):
        os.mkdir(folder)
    else:
        if not os.path.isdir(folder):
            os.mkdir(folder)


def export_img(exp_dir, img_ubyte, label_ubyte):
    """
    生成数据集
    """
    check_folder(exp_dir)
    images = decode_idx3_ubyte(img_ubyte)
    labels = decode_idx1_ubyte(label_ubyte)

    nums = len(labels)
    for i in range(nums):
        img_dir = os.path.join(exp_dir, str(labels[i]))
        check_folder(img_dir)
        img_file = os.path.join(img_dir, str(i) + '.png')
        imarr = images[i]
        cv2.imwrite(img_file, imarr)


def parser_mnist_data(data_dir):
    train_dir = os.path.join(data_dir, 'train')
    train_img_ubyte = os.path.join(data_dir, 'train-images.idx3-ubyte')
    train_label_ubyte = os.path.join(data_dir, 'train-labels.idx1-ubyte')
    export_img(train_dir, train_img_ubyte, train_label_ubyte)

    test_dir = os.path.join(data_dir, 'test')
    test_img_ubyte = os.path.join(data_dir, 'test-images.idx3-ubyte')
    test_label_ubyte = os.path.join(data_dir, 'test-labels.idx1-ubyte')
    export_img(test_dir, test_img_ubyte, test_label_ubyte)

'''
datadir = './data/'
parser_mnist_data(datadir)
'''