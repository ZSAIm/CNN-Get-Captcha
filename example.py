import urllib.request
import img_process
import matplotlib.pyplot as plt
from io import BytesIO
from PIL import Image
import numpy as np
import tensorflow as tf
import inference
from constant import *
import os

ZFsoft_CAPTCHA_URL = 'http://jwgldx.gdut.edu.cn/CheckCode.aspx'

MODEL_INDEX = 40000
MODEL_NAME_FORMAT = MODEL_NAME + '-%s'


class CaptchaReconiton():
    def __init__(self):
        with tf.Graph().as_default() as g:
            with tf.device('/cpu:0'):
                with tf.name_scope('input'):
                    self.input = tf.placeholder(tf.float32, [None, INPUT_NODE], name='x-input')
                    y_ = tf.placeholder(tf.float32, [None, OUTPUT_NODE], name='y-input')
                    self.keep_prob = tf.placeholder(tf.float32, name='keep_prob')

                x_reshape = tf.reshape(self.input, [-1, IMAGE_HEIGHT, IMAGE_WIDTH, 1], name='input-reshape')

                self.output = inference.inference(x_reshape, self.keep_prob)
                self.output_reshape = tf.reshape(self.output, [-1, NUM_CAPTCHA, LABEL_LEN])


                # correct_prediction = tf.equal(tf.argmax(self.output_reshape, 2), tf.argmax(y_, 2))
                # accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
                self.saver = tf.train.Saver()
                # with tf.Session() as sess:

                self.sess = tf.Session()
                self.saver.restore(sess=self.sess, save_path=os.path.join(MODEL_SAVE_PATH, MODEL_NAME_FORMAT % MODEL_INDEX))

    def run(self, img_array):

        predict = self.sess.run(self.output_reshape, feed_dict={self.input: img_array, self.keep_prob: 1})
        ret = []
        for i in predict:
            ret.append(argmax_onehots(i))

        return ret

    def batch_run(self):
        pass

def get_image():
    res = urllib.request.urlopen(ZFsoft_CAPTCHA_URL)
    fp = BytesIO()
    fp.write(res.read())
    return fp

def argmax_onehots(onehots):
    chars = ''
    onehots = np.reshape(onehots, [NUM_CAPTCHA, LABEL_LEN])

    for i in np.argmax(onehots, axis=1):
        chars += CHAR_SET[i]
    return chars


def main():
    img_fp = get_image()
    imgsrc = Image.open(img_fp)
    img = imgsrc.convert('L')
    imgpx = img.load()

    img_process.binary(img, imgpx, 125)
    img_process.clear_noise(img, imgpx)

    img_array = np.array(img)

    img_array_reshape = np.reshape(img_array, [1, IMAGE_HEIGHT * IMAGE_WIDTH])
    print(exm.run(img_array_reshape))

    plt.imshow(img_array)

exm = CaptchaReconiton()


if __name__ == '__main__':
    main()
    # import batching
    # for i, j in batching.batch_op():






