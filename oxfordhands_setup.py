#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Tue Jan 23 13:49:24 2018

@author: GustavZ
"""
import os
import tarfile
import six.moves.urllib as urllib
import shutil as sh

def create_directory(dir_path):
    if not os.path.exists(dir_path):
        os.makedirs(dir_path)

def download_dataset(dataset_name, dataset_url, tarfile_path):
    if not os.path.exists(tarfile_path):
        print(
            "> downloading Oxford Hand dataset (250MB)")
        opener = urllib.request.URLopener()
        opener.retrieve(dataset_url, tarfile_path)
        print("> download complete")
        extract_files(dataset_name, tarfile_path)
    else:
        print("> Oxford Hand dataset already downloaded.\sttart cleaning structure")
        extract_files(dataset_name, tarfile_path)
            
def extract_files(dataset_name, tarfile_path):
        if not os.path.exists(dataset_name):
            tar = tarfile.open(tarfile_path)
            print("> Extracting Dataset files")
            tar.extractall()
            print("> Extraction complete")
            tar.close()

def rename_double(path,name):
    if os.path.isfile(path+name):
        newname = 'x'+ name
        print('> {} already exists\
              \n> renaming to {}!'.format(name,newname))
        rename_double(path,newname)
    else:
        return name

def check_equal(src_dir, drc_dir):
    src = len([name for name in os.listdir(src_dir) if os.path.isfile(name)])
    drc = len([name for name in os.listdir(drc_dir) if os.path.isfile(name)])
    if src == drc:
        print("> equal directory sizes, everything ok!")
        return True
    else:
        print("> unequal directory sizes, manual check necessary!")
        return False

def create_label_map():
    label_map = "data/label_map.pbtxt"
    if not os.path.isfile(label_map):
        f = open(label_map,"w")
        f.write("item {\n  id: 1\n  name: 'hand'\n}")
        f.close()
    print("> created {}".format(label_map))

def cleanup_structure(data_path, dataset_path, tarfile_path):
    check = []
    create_directory(data_path)
    print('> merge training and vildation set\
          \n  and copy all files to data/ directory')

    for directory in ['test','validation','training']:
        for typ in ['images','annotations']:
            src_dir = dataset_path + '/{}_dataset/{}_data/{}/'.format(directory,directory,typ)

            if directory is 'test':
                if typ is 'annotations':
                    drc_dir = data_path+'eval/{}/mat/'.format(typ)
                else:
                    drc_dir = data_path+'eval/{}/'.format(typ)
            else:
                if typ is 'annotations':
                    drc_dir = data_path+'train/{}/mat/'.format(typ)
                else:
                    drc_dir = data_path+'train/{}/'.format(typ)

            create_directory(drc_dir)
            for file in os.listdir(src_dir):
                if file.endswith(".jpg") or file.endswith(".mat"):
                    newfile = rename_double(drc_dir,file)
                    sh.copyfile(src_dir+file, drc_dir+newfile)
                    print (newfile)
            check.append(check_equal(src_dir, drc_dir))
    if not False in check:
        sh.rmtree(dataset_path)
        #os.remove(tarfile_path)
        print('> Dataset successuflly set up!')
    else:
        print("> check manually for possible errors in created /data directory!")



def main():
    CWD = os.getcwd()
    dataset_name = 'hand_dataset'
    tarfile_path = CWD+'/hand_dataset.tar.gz'
    dataset_url = 'http://www.robots.ox.ac.uk/~vgg/data/hands/downloads/hand_dataset.tar.gz'
    dataset_path = CWD+'/'+dataset_name
    data_path = CWD+'/data/'

    download_dataset(dataset_name, dataset_url, tarfile_path)
    cleanup_structure(data_path, dataset_path, tarfile_path)
    create_label_map()


if __name__ == '__main__':
    main()
