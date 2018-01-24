# deeptraining_hands

This is a collection of scripts to setup datasets and/or to convert files of certain boundingbox-annotation formats into another <br />
Run one or several scripts to get to your desired training format. <br />
> Note: Some of the scripts use packages provided by [tensorflow's API](https://github.com/tensorflow/models/tree/master/research/object_detection) <br />
So make sure to include `tensorflow/models/research` to your PYTHONPATH

To Set Up the Oxford- and/or Egohands Dataset, run:
```
egohands_setup.py
oxfordhands_setup.py
```
Example: to get from `.mat` annotations to a tensorflow runnable `.record` file, run:
```
mat_to_xml.py
xml_to_csv.py
csv_img_to_tfrecord.py
```

All scripts create/support following folder structure to be able to support tensorflow aswell as yolo-darknet projects:
```
.
├── data
│   ├── train 
│   │   ├── annotations
│   │   │   ├── mat
│   │   │   │   ├──file1.mat
│   │   │   │   └── ...
│   │   │   └── xml
│   │   │       ├──file1.xml
│   │   │       └── ...
│   │   ├── labels
│   │   │   ├──file1.txt
│   │   │   └── ...
│   │   └── images
│   │       ├──file1.jpg
│   │       └── ...
│   ├── eval
│   │   └── ...
│   │
│   ├── train_labels.csv
│   ├── eval_labels.csv
│   ├── label_map.pbtxt
│   ├── train.record
│   ├── eval.record
│   ├── train.txt
│   └── eval.txt
│   
└── model
```
See ```howto_tf``` and ```howto_yolo``` for information how to train on yolo or tensorflow
