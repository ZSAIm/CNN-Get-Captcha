"""
**********************************************************************
**********************************************************************
**      author: ZSAIm
**      email:  405935987@163.com
**      github: https://github.com/ZSAIm/CaptchaReconition-CNN
**
**                                          programming by python 3.5
**
**                                                      9.9-2018
**********************************************************************
**********************************************************************
"""

import random
from wheezy.captcha.image import captcha, background, noise, rotate, text, curve, warp, offset, smooth
from constant import *
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
import img_process
from io import BytesIO
import threading
import os
import time

FONT_PATH = 'arialbd.ttf'

THRESHOLD = 170

NUM_SHARDS = 5
INSTANCES_PER_SHARD = 10000

global_count = 0
count_lock = threading.Lock()

def random_chars(num):
    chars = ''
    for i in range(num):
        chars += CHAR_SET[random.randint(0, len(CHAR_SET) - 1)]
    return chars

def generate_image(num):
    captcha_model = captcha(width=IMAGE_WIDTH, height=IMAGE_HEIGHT, drawings=[
        background(color='#FFFFFF'),
        text(font_sizes=[19, 20, 21, 22],
             fonts=[FONT_PATH],
             drawings=[
                 rotate(angle=15),
                 offset(0.10, 0.10)
             ],
             start_x=0,
             start_y=0,
             squeeze_factor=0.88),
    ])

    label = random_chars(num)

    imgsrc = captcha_model(label)

    image = imgsrc.convert('L')
    imgpx = image.load()

    img_process.binary(image, imgpx, THRESHOLD + random.randint(-20, 15))
    img_process.clear_noise(image, imgpx)

    # image_array = np.array(image)
    # plt.imshow(image_array)
    fp = BytesIO()
    image.save(fp, 'JPEG')
    return fp.getvalue(), label

def batch_dump():
    threads = []
    max_thread = 20

    for i in range(NUM_SHARDS):

        while True:
            threads = list(filter(lambda x: x.isAlive() is True, threads))
            time.sleep(0.2)
            if global_count % 1000 == 0:
                print(global_count)
            if len(threads) < max_thread:
                break

        thd = threading.Thread(target=image_dump, args=(i,))
        threads.append(thd)
        thd.start()

def image_dump(index_shard):
    global global_count, count_lock
    tf_writer = tf.python_io.TFRecordWriter(os.path.join(DATA_SET_PATH,
                                                         TFRECORD_NAME % (index_shard, NUM_SHARDS)))
    for i in range(INSTANCES_PER_SHARD):
        raw_img, label_str = generate_image(4)
        int_label = 0

        for j, k in enumerate(label_str):
            int_label += CHAR_SET.index(k) * (len(CHAR_SET) ** j)

        example = tf.train.Example(features=tf.train.Features(
                feature={
                    'int_label': tf.train.Feature(int64_list=tf.train.Int64List(value=[int_label])),
                    'raw_image': tf.train.Feature(bytes_list=tf.train.BytesList(value=[raw_img])),
                }))

        tf_writer.write(example.SerializeToString())
        with count_lock:
            global_count += 1

    tf_writer.close()


if __name__ == '__main__':
    batch_dump()
