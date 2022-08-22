## Introduction
A lot of wonderful datasets are now available online, such as COCO or Imagenet. These datasets help push the limits of 
computer vision. But it's not easy for us to do some small experiments with such a large number of images to quickly 
test the validity of algorithm. For this reason, I created a small dataset named "yymnist" to do both classification 
and object detection.


| classification | object detection |
|---|---
|![image](./docs/classification.png)|![image](./docs/detection.jpg)|

## Installation

Clone this repo and install some dependencies

```
$ pip install opencv-python
$ git clone https://github.com/cmu-sei/yymnist.git
```

## Classification

If you want to use this dataset for classification, use the `mnist` folder. The filename of every image starts with 
its label. For example:

```
mnist/train/0_32970.pgm -> 0
mnist/train/5_12156.pgm -> 5
```
## Objection Detection

```
$ python yymnist/make_data.py
$ python yymnist/show_image.py # [option]
```
Running `make_data.py` will produce 1000 images in the `./yymnist/Images` directory and a `label.txt` file in the 
`./yymnist/` directory. The resulting images can be used as an object detection dataset. The `label.txt` file describes 
the images that were produced. Each line of `label.txt` adheres to the following format:

```
# image_path x_min, y_min, x_max, y_max, class_id  x_min, y_min ,..., class_id 
```

`image_path` indicates the path to the image file. The remaining data describes the ground truth location(s) of any 
objects in the image. Bounding box coordinates of the object in the format (x_min, y_min, x_max, y_max) are provided, 
along with an integer indicating which class the object belongs to.

To get a reproducible data set (so others can generate the same data using the same code version) provide the `--seed` 
switch to `make_data.py` along with an unsigned integer seed value, e.g. an integer in the range 0 to 2**32-1. 
For example:

```
$ python yymnist/make_data.py --seed 1234
```


