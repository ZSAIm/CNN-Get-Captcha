import tensorflow as tf
from constant import *
import inference
import batching
import os





def train():
    """first train"""
    LEARNING_RATE_BASE = 0.001
    LEARNING_RATE_DECAY = 0.99
    # BATCH_SIZE = 50

    with tf.name_scope('input'):
        x = tf.placeholder(tf.float32, [None, INPUT_NODE], name='x-input')
        y_ = tf.placeholder(tf.float32, [None, OUTPUT_NODE], name='y-input')
        keep_prob = tf.placeholder(tf.float32, name='keep_prob')

    x_reshape = tf.reshape(x, [-1, IMAGE_HEIGHT, IMAGE_WIDTH, 1], name='input-reshape')

    output = inference.inference(x_reshape, keep_prob)

    global_step = tf.Variable(0, trainable=False)

    with tf.name_scope('loss_function'):
        cross_entropy = tf.nn.sigmoid_cross_entropy_with_logits(labels=y_, logits=output)
        loss = tf.reduce_mean(cross_entropy)

    with tf.name_scope('train_step'):
        learning_rate = tf.train.exponential_decay(LEARNING_RATE_BASE, global_step, TRAINING_STEP, LEARNING_RATE_DECAY)

        optimizer = tf.train.AdamOptimizer(learning_rate).minimize(loss, global_step=global_step)

    saver = tf.train.Saver(max_to_keep=200)

    with tf.Session(config=tf.ConfigProto(allow_soft_placement=True, log_device_placement=True)) as sess:
        tf.global_variables_initializer().run()

        for image_batch, label_batch in batching.batch_op():
            _, loss_value, step = sess.run([optimizer, loss, global_step],
                                           feed_dict={x: image_batch, y_: label_batch, keep_prob: 0.5})

            if step % 10 == 0:
                print('After %d training step(s), loss on training.' 'batch is %g' % (step, loss_value))
                if step % 200 == 0:
                    saver.save(sess, os.path.join(MODEL_SAVE_PATH, MODEL_NAME), global_step=global_step)


def reinforcement_learning():
    """base on MODEL_INDEX"""
    MODEL_INDEX = 35000
    MODEL_NAME_BASE = MODEL_NAME + '-%s' % MODEL_INDEX
    LEARNING_RATE_BASE = 0.001
    LEARNING_RATE_DECAY = 0.99
    # BATCH_SIZE = 100

    with tf.name_scope('input'):
        x = tf.placeholder(tf.float32, [None, INPUT_NODE], name='x-input')
        y_ = tf.placeholder(tf.float32, [None, OUTPUT_NODE], name='y-input')
        keep_prob = tf.placeholder(tf.float32, name='keep_prob')

    x_reshape = tf.reshape(x, [-1, IMAGE_HEIGHT, IMAGE_WIDTH, 1], name='input-reshape')

    output = inference.inference(x_reshape, keep_prob)
    global_step = tf.Variable(MODEL_INDEX, trainable=False)

    with tf.name_scope('loss_function'):
        cross_entropy = tf.nn.sigmoid_cross_entropy_with_logits(labels=y_, logits=output)
        loss = tf.reduce_mean(cross_entropy)

    with tf.name_scope('train_step'):
        learning_rate = tf.train.exponential_decay(LEARNING_RATE_BASE, global_step, TRAINING_STEP, LEARNING_RATE_DECAY)

        optimizer = tf.train.GradientDescentOptimizer(learning_rate).minimize(loss, global_step=global_step)

    saver = tf.train.Saver(max_to_keep=200)

    with tf.Session(config=tf.ConfigProto(allow_soft_placement=True, log_device_placement=True)) as sess:
        saver.restore(sess=sess, save_path=os.path.join(MODEL_SAVE_PATH, MODEL_NAME_BASE))

        for image_batch, label_batch in batching.batch_op():
            _, loss_value, step = sess.run([optimizer, loss, global_step],
                                           feed_dict={x: image_batch, y_: label_batch, keep_prob: 0.45})

            if step % 10 == 0:
                print('After %d training step(s), loss on training.' 'batch is %g' % (step, loss_value))
                if step % 200 == 0:
                    saver.save(sess, os.path.join(MODEL_SAVE_PATH, MODEL_NAME), global_step=global_step)


if __name__ == '__main__':

    # train()
    reinforcement_learning()
