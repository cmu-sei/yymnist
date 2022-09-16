#! /usr/bin/env python
# coding=utf-8
# ================================================================
#   Copyright (C) 2019 * Ltd. All rights reserved.
#
#   Editor      : VIM
#   File name   : make_data.py
#   Author      : YunYang1994
#   Created date: 2019-07-12 20:53:30
#   Description :
#
# ================================================================

import os
import cv2
import numpy as np
import shutil
import random
import sys
import json
import argparse
from pathlib import Path

# ================================================================
# Changelog
# 2022-09-15 - sei-hmmoore - Wrapped the parser into a main() function,
#                            Wrapped the logic into function outside of main,
#                            Added in ability to output coco.json format
#
# ================================================================


# 2022-09-15 - sei-hmmoore - conversion of labels.txt to coco.json
def txt_to_coco(data_root, labels_file, output_file):
    data_root = Path(data_root)
    if not data_root.exists():
        raise EnvironmentError(data_root, "Data root does not exist!")
    label_path = Path(labels_file)
    if not label_path.exists():
        raise EnvironmentError("Labels file does not exist!")

    # open file in read mode
    label = open(label_path, 'r')

    category_dict = [
        {"id": 0, "name": 'Zero'},
        {"id": 1, "name": 'One'},
        {"id": 2, "name": 'Two'},
        {"id": 3, "name": 'Three'},
        {"id": 4, "name": 'Four'},
        {"id": 5, "name": 'Five'},
        {"id": 6, "name": 'Six'},
        {"id": 7, "name": 'Seven'},
        {"id": 8, "name": 'Eight'},
        {"id": 9, "name": 'Nine'}]
    result = {'annotations': [], 'categories': category_dict, 'images': []}

    # iterate through one line/image at a time
    count = 0
    a_count = 0
    while True:
        line = label.readline()
        if not line:
            break
        sect = line.split()  # (path, box1, box2, ...)

        # ----do for each line/image-----------

        # images[]
        image_path = Path(sect[0])  # /yymnist_system_test/yymnist/Images10/000001.jpg
        # updates the path to relative. removes the data_root
        rel_image_path = image_path.relative_to("/dataroot")  # yymnist/Images10/000001.jpg
        image = {
            "id": int(count),
            "file_name": str(rel_image_path),  # same as first string in the line
            "height": 416,
            "width": 416
        }
        result['images'].append(image)

        # lienses[]

        # info[]

        # ----do for each box on the image----

        # annotations[]
        for b in range(1, len(sect)):
            numb = sect[b].split(",")  # (xmin, ymin, xmax, ymax, class)
            xmin = int(numb[0])
            ymin = int(numb[1])
            xmax = int(numb[2])
            ymax = int(numb[3])
            width = (xmax - xmin)
            height = (ymax - ymin)
            area = height * width
            x_center = (xmin + xmax) / 2
            y_center = (ymin + ymax) / 2
            box = [int(x_center), int(y_center), width, height]

            anno = {
                "id": int(a_count),
                # num annotation for this image is what i was told but looks like it is # annotations overall
                "image_id": int(count),  # link to image id
                "category_id": int(numb[4]),  # 5th num of sect. the actual letter category id
                "area": int(area),  # width * height
                "bbox": box,  # [x,y,width,height] x,y are the center coordinates
                "iscrowd": 0
            }
            result['annotations'].append(anno)
            a_count += 1
        # end for
        count += 1
    # end while

    label.close()

    # writing to the output file
    with open(output_file, 'w') as f:
        json.dump(result, f, indent=4)


# sei-amellinger 2022-08-17
# We use a global randomizer instance instead of random
randomizer = random.Random()


def generate_txt(labels_txt, images_num, images_path, image_paths, sizes, image_size):
    with open(labels_txt, "w") as wf:
        image_num = 0
        while image_num < images_num:
            image_path = os.path.realpath(os.path.join(images_path, "%06d.jpg" % (image_num + 1)))
            annotation = image_path
            blanks = np.ones(shape=[image_size, image_size, 3]) * 255
            bboxes = [[0, 0, 1, 1]]
            labels = [0]
            data = [blanks, bboxes, labels]
            bboxes_num = 0

            # small object
            ratios = [0.5, 0.8]
            # 2022-08-17 - Switched to seeded randomizer
            N = randomizer.randint(0, sizes[0])

            if N != 0: bboxes_num += 1
            for _ in range(N):
                # 2022-08-17 - Switched to seeded randomizer
                ratio = randomizer.choice(ratios)
                idx = randomizer.randint(0, 54999)

                data[0] = make_image(data, image_paths[idx], image_size, ratio)

            # medium object
            ratios = [1., 1.5, 2.]
            # 2022-08-17 - Switched to seeded randomizer
            N = randomizer.randint(0, sizes[1])
            if N != 0: bboxes_num += 1
            for _ in range(N):
                # 2022-08-17 - Switched to seeded randomizer
                ratio = randomizer.choice(ratios)
                idx = randomizer.randint(0, 54999)
                data[0] = make_image(data, image_paths[idx], image_size, ratio)

            # big object
            ratios = [3., 4.]
            # 2022-08-17 - Switched to seeded randomizer
            N = randomizer.randint(0, sizes[2])
            if N != 0: bboxes_num += 1
            for _ in range(N):
                # 2022-08-17 - Switched to seeded randomizer
                ratio = randomizer.choice(ratios)
                idx = randomizer.randint(0, 54999)
                data[0] = make_image(data, image_paths[idx], image_size, ratio)

            if bboxes_num == 0: continue
            cv2.imwrite(image_path, data[0])
            for i in range(len(labels)):
                if i == 0: continue
                xmin = str(bboxes[i][0])
                ymin = str(bboxes[i][1])
                xmax = str(bboxes[i][2])
                ymax = str(bboxes[i][3])
                class_ind = str(labels[i])
                annotation += ' ' + ','.join([xmin, ymin, xmax, ymax, str(class_ind)])
            image_num += 1
            print("=> %s" % annotation)
            wf.write(annotation + "\n")


# 2022-08-17 - added randomizer setup function
def setup_randomizer(seed):
    global randomizer
    if seed is None:
        seed = random.randint(0, 2 ** 32 - 1)
    else:
        if seed > 2 ** 32 - 1:
            print("The random seed must be and unsigned 32 bit integer. Exiting")
            sys.exit(-1)

    print(f"Using random seed: {seed}")
    randomizer.seed(seed)

def compute_iou(box1, box2):
    """xmin, ymin, xmax, ymax"""

    A1 = (box1[2] - box1[0]) * (box1[3] - box1[1])
    A2 = (box2[2] - box2[0]) * (box2[3] - box2[1])

    xmin = max(box1[0], box2[0])
    ymin = max(box1[1], box2[1])
    xmax = min(box1[2], box2[2])
    ymax = min(box1[3], box2[3])

    if ymin >= ymax or xmin >= xmax: return 0
    return ((xmax - xmin) * (ymax - ymin)) / (A1 + A2)


def make_image(data, image_path, image_size, ratio:float=1.0):
    blank = data[0]
    boxes = data[1]
    label = data[2]

    ID = image_path.split("/")[-1][0]
    image = cv2.imread(image_path)
    image = cv2.resize(image, (int(28 * ratio), int(28 * ratio)))
    h, w, c = image.shape

    while True:
        # 2022-08-17 - Switched to seeded randomizer
        # NOTE: We switched to using a python randomizer instead of numpy
        xmin = randomizer.randint(0, image_size - w)
        ymin = randomizer.randint(0, image_size - h)

        xmax = xmin + w
        ymax = ymin + h
        box = [xmin, ymin, xmax, ymax]

        iou = [compute_iou(box, b) for b in boxes]
        if max(iou) < 0.02:
            boxes.append(box)
            label.append(ID)
            break

    for i in range(w):
        for j in range(h):
            x = xmin + i
            y = ymin + j
            blank[y][x] = image[j][i]

    # cv2.rectangle(blank, (xmin, ymin), (xmax, ymax), [0, 0, 255], 2)
    return blank


# 2022-09-15 - sei-hmmoore Wrapped main logic into main() function
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--images_num", type=int, default=1000)
    parser.add_argument("--image_size", type=int, default=416)
    parser.add_argument("--images_path", type=str, default="./yymnist/Images/")
    parser.add_argument("--labels_txt", type=str, default="./yymnist/labels.txt")
    # sei-hmmoore 2022-09-15
    parser.add_argument("--coco_json", type=str, default="./yymnist/Images.json")
    parser.add_argument("--small", type=int, default=3)
    parser.add_argument("--medium", type=int, default=6)
    parser.add_argument("--big", type=int, default=3)
    # sei-amellinger 2022-08-17
    parser.add_argument("--seed", type=int, default=None,
                        help="Seed to use for randomization. Must be 32 bit unsigned for numpy. "
                             "By default, one will be made for you.")

    flags = parser.parse_args()

    setup_randomizer(flags.seed)

    SIZE = flags.image_size
    sizes = [flags.small, flags.medium, flags.big]

    if os.path.exists(flags.images_path): shutil.rmtree(flags.images_path)
    os.mkdir(flags.images_path)

    image_paths = [os.path.join(os.path.realpath("."), "./yymnist/mnist/train/" + image_name) for image_name in
                   os.listdir("./yymnist/mnist/train")]
    image_paths += [os.path.join(os.path.realpath("."), "./yymnist/mnist/test/" + image_name) for image_name in
                    os.listdir("./yymnist/mnist/test")]

    # sei-hmmoore 2022-09-15
    # Generate labels.txt
    generate_txt(flags.labels_txt, flags.images_num, flags.images_path, image_paths, sizes, SIZE)
    # sei-hmmoore 2022-09-15
    # Generate coco.json
    txt_to_coco(flags.images_path, flags.labels_txt, flags.coco_json)


if __name__ == "__main__":
    main()
