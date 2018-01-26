#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Mon Jan 15 16:15:36 2018

@author: GustavZ
"""

'''
This file converts the hand dataset (https://www.robots.ox.ac.uk/~vgg/data/hands/) to Pascal Format
'''

import numpy as np
import cv2
import scipy.io as sio
import os
from os.path import isfile, join
from xml.etree import ElementTree as et
from xml.dom import minidom


def get_img_size_node(img_path):
    img = cv2.imread(img_path)
    img_size = img.shape
    size_node = et.Element('size')

    width_node = et.Element('width')
    width_node.text = str(img_size[0])

    size_node.append(width_node)

    height_node = et.Element('height')
    height_node.text = str(img_size[1])
    size_node.append(height_node)

    depth_node = et.Element('depth')
    depth_node.text = str(img_size[2])
    size_node.append(depth_node)

    return size_node


def get_object_node(hand):

    hand_node = et.Element('object')

    name_node = et.Element('name')
    name_node.text = 'hand'
    hand_node.append(name_node)

    pose_node = et.Element('pose')
    pose_node.text = 'Unspecified'
    hand_node.append(pose_node)

    truncated_node = et.Element('truncated')
    truncated_node.text = '0'
    hand_node.append(truncated_node)

    difficult_node = et.Element('difficult')
    difficult_node.text = '0'
    hand_node.append(difficult_node)

    bbox_node = et.Element('bndbox')
    xmin_node = et.Element('xmin')
    xmin_node.text = '{0:.0f}'.format(hand[0])
    bbox_node.append(xmin_node)

    ymin_node = et.Element('ymin')
    ymin_node.text = '{0:.0f}'.format(hand[1])
    bbox_node.append(ymin_node)

    xmax_node = et.Element('xmax')
    xmax_node.text = '{0:.0f}'.format(hand[2])
    bbox_node.append(xmax_node)

    ymax_node = et.Element('ymax')
    ymax_node.text = '{0:.0f}'.format(hand[3])
    bbox_node.append(ymax_node)

    hand_node.append(bbox_node)

    return hand_node

#Read the given mat file and add hand objects to the corresponding XML file
def read_mat_file(filepath, filename,IMG_FILES_PATH, XML_FILES_PATH):

    mat_data = sio.loadmat(filepath)
    hand_pos = [] # To store the hand positions in an image
    old_pts = []
    curr_pts = []
    #For all hands in the image align the bounding box to an axis
    for i in range(len(mat_data['boxes'][0])):
        xmin = float('inf')
        ymin = float('inf')
        xmax = -1
        ymax = -1

        for j in range(4):
            y, x = mat_data['boxes'][0][i][0][0][j][0]
            curr_pts.append(make_int([x,y]))
            if xmin > x:
                xmin = x
            if ymin > y:
                ymin = y
            if xmax < x:
                xmax = x
            if ymax < y:
                ymax = y
        old_pts.append(curr_pts)
        hand_pos.insert(0, [xmin, ymin, xmax, ymax])
        
    img_filename = filename.split('.')[0] + '.jpg'
    image = IMG_FILES_PATH + '/' + img_filename
    visualize(image,old_pts,hand_pos,filename,1)

    # Create the XML file
    create_xml_file(hand_pos, filename,IMG_FILES_PATH, XML_FILES_PATH)

def create_xml_file(hand_pos, filename,IMG_FILES_PATH, XML_FILES_PATH):

    #Generate the image file name
    img_filename = filename.split('.')[0] + '.jpg'
    xml_filename = filename.split('.')[0] + '.xml'

    root = et.Element('annotation')
    tree = et.ElementTree(root) # Create a XML tree with root as 'annotation'

    #Create an element folder
    folder = et.Element('folder')
    folder.text = 'imgs/'
    root.append(folder)

    #Add filename
    filename_node = et.Element('filename')
    filename_node.text = img_filename
    root.append(filename_node)

    #Add filepath
    filepath_node = et.Element('path')
    filepath_node.text = filename_node.text
    root.append(filepath_node)

    # Node for the size of the image
    img_path = join(IMG_FILES_PATH, img_filename)
    size_node = get_img_size_node(img_path)
    root.append(size_node)

    #Add segmented node
    segmented_node = et.Element('segmented')
    segmented_node.text = '0'
    root.append(segmented_node)

    #Add the objects
    for hand in hand_pos:
        hand_node = get_object_node(hand)
        root.append(hand_node)

    rough_xml = et.tostring(root, 'utf-8')
    rough_xml = minidom.parseString(rough_xml)
    pretty_xml = rough_xml.toprettyxml()

    # Save the XML file
    xml_path = join(XML_FILES_PATH, xml_filename)
    with open(xml_path, 'w') as xml_file:
        xml_file.write(pretty_xml)
        
def visualize(image,pts,boxes,filename,time):
    font = cv2.FONT_HERSHEY_SIMPLEX
    img = cv2.imread(image)
    
    for pt in pts:
        pt = np.array(pt)
        cv2.polylines(img,[pt], True, (0, 0, 255), 2)
    for box in boxes:
        xmin, ymin, xmax, ymax = make_int(box)
        cv2.rectangle(img, (xmin, ymin), (xmax, ymax), (0, 255, 0), 2)
        
    cv2.putText(img, filename, (0,20), font, 0.75, (77, 255, 9), 2)
    cv2.imshow('label_check', img)
    cv2.waitKey(time)
    
def make_int(box):
    newbox = []
    for pt in box:
        newbox.append(int(pt))
    return newbox


def main():
    # Read a .mat file and convert it to a pascal format
    for directory in ['train','eval']:

        if not os.path.exists('data/{}/annotations/xml'.format(directory)):
            os.makedirs('data/{}/annotations/xml'.format(directory))

        MAT_FILES_PATH = os.path.join(os.getcwd(), 'data/{}/annotations/mat'.format(directory))
        XML_FILES_PATH = os.path.join(os.getcwd(), 'data/{}/annotations/xml'.format(directory))
        IMG_FILES_PATH = os.path.join(os.getcwd(), 'data/{}/images'.format(directory))

        # List all files in the MAT_FILES_PATH and ignore hidden files (.DS_STORE for Macs)
        mat_files = [[join(MAT_FILES_PATH, x), x] for x in os.listdir(MAT_FILES_PATH) if isfile(join(MAT_FILES_PATH, x)) and x[0] is not '.']
        mat_files.sort()
	
        # Iterate through all files and convert them to XML
        for mat_file in mat_files:
            #print(mat_file)
            read_mat_file(mat_file[0], mat_file[1],IMG_FILES_PATH, XML_FILES_PATH)
            #break
        print ("successfully converted {}-labels from mat to pascal-xml").format(directory)
    cv2.destroyAllWindows()

if __name__ == '__main__':
    main()
