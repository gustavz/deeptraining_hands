#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Mon Jan 22 10:37:44 2018

@author: GustavZ
"""
from __future__ import division
import pandas as pd
import os
from object_detection.utils import label_map_util

# convert pascal_voc bounding boxes to yolo_darknet format
def convert(width, height, xmin, ymin, xmax, ymax):
    dw = 1./width
    dh = 1./height
    x = (xmin + xmax)/2.0
    y = (ymin + ymax)/2.0
    w = xmax - xmin
    h = ymax - ymin
    x = round(x*dw,6)
    w = round(w*dw,6)
    y = round(y*dh,6)
    h = round(h*dh,6)
    string = str((x,y,w,h))
    string = string.replace(',', '')
    string = string.replace('(', '')
    string = string.replace(')', '')
    return string

# create a text file for each image listed in the csv file
def write(df, txt_path, img_path, label_map, directory):
    names = []
    for index, row in df.iterrows():
        name = str(row['filename'])
        name = name.replace('.jpg','')
        width = row['width']
        height = row['height']
        xmin = row['xmin']
        xmax = row['xmax']
        ymin = row['ymin']
        ymax = row['ymax']
        class_id = str(label_map[row['class']]-1)
        string = convert(width,height,xmin,ymin,xmax,ymax)
        if not name in names:
            f = open(txt_path+"/{}.txt".format(name),"w")
            f.write(class_id+" ")
            f.write(string)
            f.close()
            names.append(name)
        else:
            f = open(txt_path+"/{}.txt".format(name),"a")
            f.write("\n"+class_id+" ")
            f.write(string)
            f.close()

    # write all image paths into a txt file
    f = open('data/{}.txt'.format(directory),"w")
    for name in names:
        f.write(img_path+name+".jpg"+"\n")
    f.close()

# create yolov config files
def write_config(label_map, network,CWD) :
    # file containing necessary training paths
    num_classes = len(label_map)
    backup_path = '{}/model/yolo_backup/'.format(CWD)
    f = open('data/{}.data'.format(network),"w")
    f.write("classes = {}\
            \ntrain = {}/data/train.txt\
            \nvalid = {}/data/eval.txt\
            \nnames = {}/data/{}.names\
            \nbackup = {}".format(num_classes,CWD,CWD,CWD,network,backup_path))
    f.close()
    if not os.path.exists(backup_path):
    	os.makedirs(backup_path)
    # file containing all class names
    f = open('data/{}.names'.format(network),"w")
    for obj in label_map:
        f.write(obj+"\n")
    f.close()


def main():

    print ('###''\n''MAKE SURE TO APPEND tensorflow/models/research TO YOUR PYTHONPATH''\n''###')
    label_map = label_map_util.get_label_map_dict(os.path.join(os.getcwd(), 'data/label_map.pbtxt'))
    network = 'handsnet'
    CWD = os.getcwd()

    for directory in ['train','eval']:
        txt_path = CWD + 'data/{}/labels'.format(directory)
        csv_path = CWD + 'data/{}_labels.csv'.format(directory)
        img_path = CWD + '/data/{}/images/'.format(directory)

        if not os.path.exists(txt_path):
            os.makedirs(txt_path)

        df = pd.read_table(csv_path, sep=",")
        write(df, txt_path, img_path, label_map, directory)
        print('Successfully created the {}-Yolo-txt files'.format(directory))
    write_config(label_map,network,CWD)
    print('Successfully created the yolo-config files')

if __name__ == '__main__':
    main()
