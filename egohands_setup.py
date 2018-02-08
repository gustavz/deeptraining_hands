"""
THIS CODE IS TAKEN FROM VICTOR DIBIA WHO ALSO WORKED ON THE SAME TOPIC
UNFORTUNATELY 2 MONTHS BEFORE I HAD THE IDEA ;)
BUT THIS PEACE OF CODE HERE IS PERFECT SO HANDS DOWN
ALL I DID WAS ALTERING IT A BIT TO MY NEEDS

SEE HIS REPO:
https://github.com/victordibia/handtracking
"""
import scipy.io as sio
import numpy as np
import os
import gc
import cv2
import time
import xml.etree.cElementTree as ET
import random
import shutil as sh
from shutil import copyfile
import zipfile
import six.moves.urllib as urllib
import csv


def save_csv(csv_path, csv_content):
    with open(csv_path, 'w') as csvfile:
        wr = csv.writer(csvfile)
        for i in range(len(csv_content)):
            wr.writerow(csv_content[i])


def get_bbox_visualize(base_path, dir):
    image_path_array = []
    for root, dirs, filenames in sorted(os.walk(base_path + dir)):
        for f in filenames:
            if(f.split(".")[1] == "jpg"):
                img_path = base_path + dir + "/" + f
                image_path_array.append(img_path)

    image_path_array.sort()
    boxes = sio.loadmat(base_path + dir + "/polygons.mat")
    # there are 100 of these per folder in the egohands dataset
    polygons = boxes["polygons"][0]
    # first = polygons[0]
    # print(len(first))
    pointindex = 0

    for first in polygons:

        font = cv2.FONT_HERSHEY_SIMPLEX

        img_id = image_path_array[pointindex]
        img = cv2.imread(img_id)

        img_params = {}
        img_params["width"] = np.size(img, 1)
        img_params["height"] = np.size(img, 0)
        head, tail = os.path.split(img_id)
        img_params["filename"] = tail
        img_params["path"] = os.path.abspath(img_id)
        img_params["type"] = "train"
        pointindex += 1

        boxarray = []
        csvholder = []
        for pointlist in first:
            pst = np.empty((0, 2), int)
            max_x = max_y = min_x = min_y = 0

            findex = 0
            for point in pointlist:
                if(len(point) == 2):
                    x = int(point[0])
                    y = int(point[1])

                    if(findex == 0):
                        min_x = x
                        min_y = y
                    findex += 1
                    max_x = x if (x > max_x) else max_x
                    min_x = x if (x < min_x) else min_x
                    max_y = y if (y > max_y) else max_y
                    min_y = y if (y < min_y) else min_y
                    # print(index, "====", len(point))
                    appeno = np.array([[x, y]])
                    pst = np.append(pst, appeno, axis=0)
                    cv2.putText(img, ".", (x, y), font, 0.7,
                                (255, 255, 255), 2, cv2.LINE_AA)

            hold = {}
            hold['minx'] = min_x
            hold['miny'] = min_y
            hold['maxx'] = max_x
            hold['maxy'] = max_y
            if (min_x > 0 and min_y > 0 and max_x > 0 and max_y > 0):
                boxarray.append(hold)
                labelrow = [tail,
                            np.size(img, 1), np.size(img, 0), "hand", min_x, min_y, max_x, max_y]
                csvholder.append(labelrow)

            cv2.polylines(img, [pst], True, (0, 255, 255), 1)
            cv2.rectangle(img, (min_x, max_y),
                          (max_x, min_y), (0, 255, 0), 1)

        csv_path = img_id.split(".")[0]
        if not os.path.exists(csv_path + ".csv"):
            cv2.putText(img, "DIR : " + dir + " - " + tail, (20, 50),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.75, (77, 255, 9), 2)
            cv2.imshow('Verifying annotation ', img)
            save_csv(csv_path + ".csv", csvholder)
            print("===== saving csv file for ", tail)
        cv2.waitKey(1) # Change this to 1000 to see every single frame


def create_directory(dir_path):
    if not os.path.exists(dir_path):
        os.makedirs(dir_path)

# combine all individual csv files for each image into a single csv file per folder.
def generate_label_files(image_dir):
    header = ['filename', 'width', 'height',
              'class', 'xmin', 'ymin', 'xmax', 'ymax']
    for root, dirs, filenames in os.walk(image_dir):
        for dir in dirs:
            csvholder = []
            csvholder.append(header)
            loop_index = 0
            for f in os.listdir(image_dir + dir):
                if(f.split(".")[1] == "csv"):
                    loop_index += 1
                    #print(loop_index, f)
                    csv_file = open(image_dir + dir + "/" + f, 'r')
                    reader = csv.reader(csv_file)
                    for row in reader:
                        csvholder.append(row)
                    csv_file.close()
                    os.remove(image_dir + dir + "/" + f)
            save_csv(image_dir + dir + "_labels.csv", csvholder)
            print("Saved label csv for ", dir, image_dir +
                  dir + "/" + dir + "_labels.csv")


# Split data, copy to train/test folders
def split_data_test_eval_train(image_dir):
    create_directory("data")
    create_directory("data/train")
    create_directory("data/eval")
    
    loop_index = 0
    """
    data_size = 4000
    data_sampsize = int(0.1 * data_size)
    random.seed(1)
    test_samp_array = random.sample(range(data_size), k=data_sampsize)
    """

    for root, dirs, filenames in os.walk(image_dir):
        for dir in dirs:
            for f in os.listdir(image_dir + dir):
                if(f.split(".")[1] == "jpg"):
                    loop_index += 1
                    #print('DEBUG: loop_index, f',loop_index, f)
                    #print('DEBUG: f.split(".")[0]',f.split(".")[0])

                    #if loop_index in test_samp_array:
                    if not np.mod(loop_index,10): 
                        os.rename(image_dir + dir +
                                  "/" + f, "data/eval/" + f)
                        os.rename(image_dir + dir +
                                  "/" + f.split(".")[0] + ".csv", "data/eval/" + f.split(".")[0] + ".csv")
                    else:
                        os.rename(image_dir + dir +
                                  "/" + f, "data/train/" + f)
                        os.rename(image_dir + dir +
                                  "/" + f.split(".")[0] + ".csv", "data/train/" + f.split(".")[0] + ".csv")
                    print(loop_index, image_dir + f)
            print(">   done scanning director ", dir)
            os.remove(image_dir + dir + "/polygons.mat")
            os.rmdir(image_dir + dir)

        print("Train/Eval content generation complete!")
        generate_label_files("data/")


def generate_csv_files(image_dir):
    for root, dirs, filenames in os.walk(image_dir):
        for dir in dirs:
            get_bbox_visualize(image_dir, dir)

    print("CSV generation complete!\nGenerating train/eval folders")
    split_data_test_eval_train("egohands/_LABELLED_SAMPLES/")


# rename image files so we can have them all in a train/test/eval folder.
def rename_files(image_dir):
    print("Renaming files")
    loop_index = 0
    for root, dirs, filenames in sorted(os.walk(image_dir)):
        for dir in dirs:
            for f in os.listdir(image_dir + dir):
                if (dir not in f):
                    if(f.split(".")[1] == "jpg"):
                        loop_index += 1
                        old = image_dir + dir + "/" + f
                        new = image_dir + dir + "/" + dir + "_" + f
                        os.rename(old, new)
                else:
                    break

    generate_csv_files("egohands/_LABELLED_SAMPLES/")

def extract_folder(dataset_path):
    if not os.path.exists("egohands"):
        zip_ref = zipfile.ZipFile(dataset_path, 'r')
        print("> Extracting Dataset files")
        zip_ref.extractall("egohands")
        print("> Extraction complete")
        zip_ref.close()
        rename_files("egohands/_LABELLED_SAMPLES/")
        
def download_egohands_dataset(dataset_url, dataset_path):
    print("\nTHIS CODE IS BASED ON VICTOR DIBIAs WORK\
          \nSEE HIS REPO:\
          \nhttps://github.com/victordibia/handtracking\n")

    is_downloaded = os.path.exists(dataset_path)
    if not is_downloaded:
        print(
            "> downloading Egohands dataset (1.3GB)")
        opener = urllib.request.URLopener()
        opener.retrieve(dataset_url, dataset_path)
        print("> download complete")
        extract_folder(dataset_path)
    else:
        print("Egohands dataset already downloaded.\nGenerating CSV files")
        extract_folder(dataset_path)
        
def create_label_map():
    label_map = "data/label_map.pbtxt"
    if not os.path.isfile(label_map):
        f = open(label_map,"w")
        f.write("item {\n  id: 1\n  name: 'hand'\n}")
        f.close()
    print("> created ",label_map)

def final_finish():
    cwd = os.getcwd()
    for directory in ['train','eval']:
        src_dir = cwd+'/data/{}/'.format(directory)
        drc_dir = cwd+'/data/{}/images/'.format(directory)
        create_directory(drc_dir)
        for file in os.listdir(src_dir):
            if file.endswith(".jpg"):
               sh.move(src_dir+file,drc_dir+file)
    sh.rmtree('egohands')
    #os.remove(EGO_HANDS_FILE)
    print('\n> creating the dataset complete\
          \n> you can now start training\
          \n> see howto_wiki for more information')
    
    
def main():
    EGOHANDS_DATASET_URL = "http://vision.soic.indiana.edu/egohands_files/egohands_data.zip"
    EGO_HANDS_FILE = "egohands_data.zip"
    
    download_egohands_dataset(EGOHANDS_DATASET_URL, EGO_HANDS_FILE)
    create_label_map()
    final_finish()
    
    
if __name__ == '__main__':
    main()
