# deeptraining_hands

Each of the provided scripts converts one boundingbox annotation format into another.
Run one or several scripts to get to your desired training format.

    to get from .mat annotations to a tensorflow runnable .record file run:
    mat_to_xml.py
    xml_to_csv.py
    csv_img_to_tfrecord.py

All scripts create/support following folder structure:
    .
    ├── data
    │   └── train 
    |       ├── annotations
    |           ├── mat
    |           ├── xml
    |           └── txt
    |       ├── images
    │   ├── eval
    |       ├── ...
    │   
    └── model
