#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Mon Jan 15 16:15:36 2018

@author: GustavZ
"""

from __future__ import division
from __future__ import print_function
from __future__ import absolute_import

import os
import io
import pandas as pd
import tensorflow as tf
from PIL import Image
from object_detection.utils import dataset_util,label_map_util
from collections import namedtuple, OrderedDict


def split(df, group):
    data = namedtuple('data', ['filename', 'object'])
    gb = df.groupby(group)
    return [data(filename, gb.get_group(x)) for filename, x in zip(gb.groups.keys(), gb.groups)]


def create_tf_example(group, path, label_map_dict):
    with tf.gfile.GFile(os.path.join(path, '{}'.format(group.filename)), 'rb') as fid:
        encoded_jpg = fid.read()
    encoded_jpg_io = io.BytesIO(encoded_jpg)
    image = Image.open(encoded_jpg_io)
    width, height = image.size

    filename = group.filename.encode('utf8')
    image_format = b'jpg'
    xmins = []
    xmaxs = []
    ymins = []
    ymaxs = []
    classes_text = []
    classes = []

    for index, obj in group.object.iterrows():
        xmins.append(obj['xmin'] / width)
        xmaxs.append(obj['xmax'] / width)
        ymins.append(obj['ymin'] / height)
        ymaxs.append(obj['ymax'] / height)
        classes_text.append(obj['class'].encode('utf8'))
        classes.append(label_map_dict[obj['class']])
    tf_example = tf.train.Example(features=tf.train.Features(feature={
        'image/height': dataset_util.int64_feature(height),
        'image/width': dataset_util.int64_feature(width),
        'image/filename': dataset_util.bytes_feature(filename),
        'image/source_id': dataset_util.bytes_feature(filename),
        'image/encoded': dataset_util.bytes_feature(encoded_jpg),
        'image/format': dataset_util.bytes_feature(image_format),
        'image/object/bbox/xmin': dataset_util.float_list_feature(xmins),
        'image/object/bbox/xmax': dataset_util.float_list_feature(xmaxs),
        'image/object/bbox/ymin': dataset_util.float_list_feature(ymins),
        'image/object/bbox/ymax': dataset_util.float_list_feature(ymaxs),
        'image/object/class/text': dataset_util.bytes_list_feature(classes_text),
        'image/object/class/label': dataset_util.int64_list_feature(classes),
    }))
    return tf_example


def main():
    for directory in ['train','eval']:

        image_path = os.path.join(os.getcwd(), 'data/{}/images/'.format(directory))
        csv_path = os.path.join(os.getcwd(), 'data/{}_labels.csv'.format(directory,directory))
        output_path = os.path.join(os.getcwd(), 'data/{}.record'.format(directory))
        label_map_dict = label_map_util.get_label_map_dict(os.path.join(os.getcwd(), 'data/label_map.pbtxt'))

        writer = tf.python_io.TFRecordWriter(output_path)
        examples = pd.read_csv(csv_path)
        grouped = split(examples, 'filename')
        for group in grouped:
            tf_example = create_tf_example(group, image_path, label_map_dict)
            writer.write(tf_example.SerializeToString())

        writer.close()
        print('Successfully created the {}-TFRecords'.format(directory))


if __name__ == '__main__':
    main()
