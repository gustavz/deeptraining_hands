"""
Created on Mon Jan 15 16:15:36 2018

@author: GustavZ
"""
import os
import glob
import pandas as pd
import xml.etree.ElementTree as ET
import cv2


def xml_to_csv(xml_path,img_path):
    xml_list = []
    for xml_file in glob.glob(xml_path + '/*.xml'):
        tree = ET.parse(xml_file)
        root = tree.getroot()
        boxes = []
        for member in root.findall('object'):
            value = (root.find('filename').text,
                     int(root.find('size')[0].text),
                     int(root.find('size')[1].text),
                     member[0].text,
                     int(member[4][0].text),
                     int(member[4][1].text),
                     int(member[4][2].text),
                     int(member[4][3].text)
                     )
            xml_list.append(value)
            boxes.append(list(value[4:]))
            name = str(value[0])
          
        image = img_path + '/' + name
        visualize(image,boxes,name,1)
    
    column_name = ['filename', 'width', 'height', 'class', 'xmin', 'ymin', 'xmax', 'ymax']
    xml_df = pd.DataFrame(xml_list, columns=column_name)
    return xml_df

def visualize(image,boxes,text,time):
    font = cv2.FONT_HERSHEY_SIMPLEX
    img = cv2.imread(image)
    
    for box in boxes:
        xmin, ymin, xmax, ymax = box
        cv2.rectangle(img, (xmin, ymin), (xmax, ymax), (0, 255, 0), 2)
        
    cv2.putText(img, text, (0,20), font, 0.75, (77, 255, 9), 2)
    cv2.imshow('label_check', img)
    cv2.waitKey(time)

def main():
    for directory in ['train','eval']:
        xml_path = os.path.join(os.getcwd(), 'data/{}/annotations/xml'.format(directory))
        img_path = os.path.join(os.getcwd(), 'data/{}/images'.format(directory))
        xml_df = xml_to_csv(xml_path,img_path)
        xml_df.to_csv('data/{}_labels.csv'.format(directory,directory), index=None)
        print('Successfully converted {} xml to csv.').format(directory)
        cv2.destroyAllWindows()


if __name__ == '__main__':
    main()
