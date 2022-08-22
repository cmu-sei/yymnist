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
import argparse
import sys


# ================================================================
# Changelog
# 2022-08-17 - sei-amellinger - Exposed setting the seed value via argparse.

# ================================================================

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


# ================================================================


parser = argparse.ArgumentParser()
parser.add_argument("--images_num", type=int, default=1000)
parser.add_argument("--image_size", type=int, default=416)
parser.add_argument("--images_path", type=str, default="./yymnist/Images/")
parser.add_argument("--labels_txt", type=str, default="./yymnist/labels.txt")
parser.add_argument("--small", type=int, default=3)
parser.add_argument("--medium", type=int, default=6)
parser.add_argument("--big", type=int, default=3)
# sei-amellinger 2022-08-17
parser.add_argument("--seed", type=int, default=None,
                    help="Seed to use for randomization. Must be 32 bit unsigned for numpy. "
                         "By default, one will be made for you.")

flags = parser.parse_args()

# sei-amellinger 2022-08-17
# We use a global randomizer instance instead of random
randomizer = random.Random()
setup_randomizer(flags.seed)

SIZE = flags.image_size

if os.path.exists(flags.images_path): shutil.rmtree(flags.images_path)
os.mkdir(flags.images_path)

image_paths = [os.path.join(os.path.realpath("."), "./yymnist/mnist/train/" + image_name) for image_name in
               os.listdir("./yymnist/mnist/train")]
image_paths += [os.path.join(os.path.realpath("."), "./yymnist/mnist/test/" + image_name) for image_name in
                os.listdir("./yymnist/mnist/test")]


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


def make_image(data, image_path, ratio:float=1.0):
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
        xmin = randomizer.randint(0, SIZE - w)
        ymin = randomizer.randint(0, SIZE - h)

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


with open(flags.labels_txt, "w") as wf:
    image_num = 0
    while image_num < flags.images_num:
        image_path = os.path.realpath(os.path.join(flags.images_path, "%06d.jpg" % (image_num + 1)))
        annotation = image_path
        blanks = np.ones(shape=[SIZE, SIZE, 3]) * 255
        bboxes = [[0, 0, 1, 1]]
        labels = [0]
        data = [blanks, bboxes, labels]
        bboxes_num = 0

        # small object
        ratios = [0.5, 0.8]
        # 2022-08-17 - Switched to seeded randomizer
        N = randomizer.randint(0, flags.small)

        if N != 0: bboxes_num += 1
        for _ in range(N):
            # 2022-08-17 - Switched to seeded randomizer
            ratio = randomizer.choice(ratios)
            idx = randomizer.randint(0, 54999)

            data[0] = make_image(data, image_paths[idx], ratio)

        # medium object
        ratios = [1., 1.5, 2.]
        # 2022-08-17 - Switched to seeded randomizer
        N = randomizer.randint(0, flags.medium)
        if N != 0: bboxes_num += 1
        for _ in range(N):
            # 2022-08-17 - Switched to seeded randomizer
            ratio = randomizer.choice(ratios)
            idx = randomizer.randint(0, 54999)
            data[0] = make_image(data, image_paths[idx], ratio)

        # big object
        ratios = [3., 4.]
        # 2022-08-17 - Switched to seeded randomizer
        N = randomizer.randint(0, flags.big)
        if N != 0: bboxes_num += 1
        for _ in range(N):
            # 2022-08-17 - Switched to seeded randomizer
            ratio = randomizer.choice(ratios)
            idx = randomizer.randint(0, 54999)
            data[0] = make_image(data, image_paths[idx], ratio)

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
