# deeptraining_hands

Each of the provided scripts converts files of a certain boundingbox annotation format into another. <br />
Run one or several scripts to get to your desired training format. <br />
Some of the scripts use packages provided by [tensorflow's API](https://github.com/tensorflow/models/tree/master/research/object_detection) so make to include `tensorflow/models/research` to your PYTHONPATH

> Example: to get from `.mat` annotations to a tensorflow runnable `.record` file, run:
```
mat_to_xml.py
xml_to_csv.py
csv_img_to_tfrecord.py
```

All scripts create/support following folder structure:
```
.
├── data
|   ├── train_labels.csv
|   ├── eval_labels.csv
|   ├── label_map.pbtxt
|   ├── train.record
|   ├── eval.record
│   └── train 
|       ├── annotations
|           ├── mat
|               ├──file1.mat
|               └── ...
|           ├── xml
|               ├──file1.xml
|               └── ...
|           └── txt
|               ├──file1.txt
|               └── ...
|       └── images
|               ├──file1.jpg
|               └── ...
│   ├── eval
|       └── ...
│   
└── model
```
