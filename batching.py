from constant import *
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import os

CAPACITY = 1000 + 3 * BATCH_SIZE


def IntlabelToOnehots_batch(int_label_batch):
    onehots_batch = np.zeros(NUM_CAPTCHA * LABEL_LEN * BATCH_SIZE)
    for i in range(BATCH_SIZE):
        for j in range(NUM_CAPTCHA):
            next_digit = int(int_label_batch[i] / LABEL_LEN ** j)
            cur_digit = next_digit % LABEL_LEN
            onehots_batch[i * NUM_CAPTCHA * LABEL_LEN + j * LABEL_LEN + cur_digit] = 1

    onehots_batch = np.reshape(onehots_batch, [BATCH_SIZE, -1])
    return onehots_batch

def OnehotToChars(onehots):
    chars = ''
    for i in range(NUM_CAPTCHA):
        for j in range(LABEL_LEN):
            if onehots[LABEL_LEN * i + j] == 1:
                chars += CHAR_SET[j]
                break
    return chars


def batch_op():
    with tf.Graph().as_default() as g:
        files = tf.train.match_filenames_once(os.path.join(DATA_SET_PATH, 'data.tfrecord-*'))

        # print(os.path.join(DATA_SET_PATH, 'data.tfrecord-*'))
        filename_queue = tf.train.string_input_producer(files, shuffle=True)

        tf_reader = tf.TFRecordReader()
        _, serializer_example = tf_reader.read(filename_queue)

        features = tf.parse_single_example(serializer_example,
                                           features={
                                               'int_label': tf.FixedLenFeature([], tf.int64),
                                               'raw_image': tf.FixedLenFeature([], tf.string),
                                           })

        image = tf.image.decode_jpeg(features['raw_image'], channels=1)
        label = tf.cast(features['int_label'], tf.int32)

        image_reshape = tf.reshape(image, [IMAGE_HEIGHT * IMAGE_WIDTH])

        image_batch, label_batch = tf.train.shuffle_batch([image_reshape, label], batch_size=BATCH_SIZE,
                                                          min_after_dequeue=100, capacity=CAPACITY, num_threads=2)

        with tf.Session() as sess:
            tf.local_variables_initializer().run()

            coord = tf.train.Coordinator()

            threads = tf.train.start_queue_runners(sess=sess, coord=coord)

            for i in range(TRAINING_STEP):
                cur_image_batch, cur_label_batch = sess.run([image_batch, label_batch])
                # cur_image_batch, cur_label_batch = sess.run([image_reshape, label])
                cur_label_batch = IntlabelToOnehots_batch(cur_label_batch)
                yield cur_image_batch, cur_label_batch

            coord.request_stop()
            coord.join(threads)

if __name__ == '__main__':
    for i, j in batch_op():

        pass