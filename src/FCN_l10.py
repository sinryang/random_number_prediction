#-*- coding: utf-8 -*-

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import math

import tensorflow as tf

from datasets.pension_lottery import datasets

N_INPUT_DATA = 10
N_TARGET_NUM = 35

NUM_INPUTS = N_INPUT_DATA * (N_TARGET_NUM + 8)

N_HL1_UNITS = NUM_INPUTS
N_HL2_UNITS = NUM_INPUTS
N_HL3_UNITS = NUM_INPUTS
N_HL4_UNITS = NUM_INPUTS
N_HL5_UNITS = NUM_INPUTS
N_HL6_UNITS = NUM_INPUTS
N_HL7_UNITS = NUM_INPUTS
N_HL8_UNITS = NUM_INPUTS
N_HL9_UNITS = NUM_INPUTS
N_HL10_UNITS = N_TARGET_NUM

epoch = 10

def main(_):
    data = datasets.load_pension_lottery_fcn_datasets(n_data=N_INPUT_DATA, shuffle=True, ratio=[0.7, None, 0.3])

    train = data.train
    test = data.test

    # Neural Network
    # Input
    X = tf.placeholder(tf.float32, [None, NUM_INPUTS])

    # Layer_1
    w_1 = tf.Variable(tf.truncated_normal([NUM_INPUTS, N_HL1_UNITS], stddev=1.0 / math.sqrt(float(NUM_INPUTS))), name='weights_1')
    b_1 = tf.Variable(tf.ones([N_HL1_UNITS]), name='biases_1')
    l_1 = tf.nn.relu(tf.matmul(X, w_1) + b_1)

    # Layer_2
    w_2 = tf.Variable(tf.truncated_normal([N_HL1_UNITS, N_HL2_UNITS], stddev=1.0 / math.sqrt(float(N_HL1_UNITS))), name='weights_2')
    b_2 = tf.Variable(tf.ones([N_HL2_UNITS]), name='biases_2')
    l_2 = tf.nn.relu(tf.matmul(l_1, w_2) + b_2)

    # Layer_3
    w_3 = tf.Variable(tf.truncated_normal([N_HL2_UNITS, N_HL3_UNITS], stddev=1.0 / math.sqrt(float(N_HL2_UNITS))), name='weights_3')
    b_3 = tf.Variable(tf.ones([N_HL3_UNITS]), name='biases_3')
    l_3 = tf.nn.relu(tf.matmul(l_2, w_3) + b_3)

    # Layer_4
    w_4 = tf.Variable(tf.truncated_normal([N_HL3_UNITS, N_HL4_UNITS], stddev=1.0 / math.sqrt(float(N_HL3_UNITS))), name='weights_4')
    b_4 = tf.Variable(tf.ones([N_HL4_UNITS]), name='biases_4')
    l_4 = tf.nn.relu(tf.matmul(l_3, w_4) + b_4)

    # Layer_5
    w_5 = tf.Variable(tf.truncated_normal([N_HL4_UNITS, N_HL5_UNITS], stddev=1.0 / math.sqrt(float(N_HL4_UNITS))), name='weights_5')
    b_5 = tf.Variable(tf.ones([N_HL5_UNITS]), name='biases_5')
    l_5 = tf.nn.relu(tf.matmul(l_4, w_5) + b_5)

    # Layer_6
    w_6 = tf.Variable(tf.truncated_normal([N_HL5_UNITS, N_HL6_UNITS], stddev=1.0 / math.sqrt(float(N_HL5_UNITS))), name='weights_6')
    b_6 = tf.Variable(tf.ones([N_HL6_UNITS]), name='biases_6')
    l_6 = tf.nn.relu(tf.matmul(l_5, w_6) + b_6)

    # Layer_7
    w_7 = tf.Variable(tf.truncated_normal([N_HL6_UNITS, N_HL7_UNITS], stddev=1.0 / math.sqrt(float(N_HL6_UNITS))), name='weights_7')
    b_7 = tf.Variable(tf.ones([N_HL7_UNITS]), name='biases_7')
    l_7 = tf.nn.relu(tf.matmul(l_6, w_7) + b_7)

    # Layer_8
    w_8 = tf.Variable(tf.truncated_normal([N_HL7_UNITS, N_HL8_UNITS], stddev=1.0 / math.sqrt(float(N_HL7_UNITS))), name='weights_8')
    b_8 = tf.Variable(tf.ones([N_HL8_UNITS]), name='biases_8')
    l_8 = tf.nn.relu(tf.matmul(l_7, w_8) + b_8)

    # Layer_9
    w_9 = tf.Variable(tf.truncated_normal([N_HL8_UNITS, N_HL9_UNITS], stddev=1.0 / math.sqrt(float(N_HL8_UNITS))), name='weights_9')
    b_9 = tf.Variable(tf.ones([N_HL9_UNITS]), name='biases_9')
    l_9 = tf.nn.relu(tf.matmul(l_8, w_9) + b_9)

    # Layer_10
    w_10 = tf.Variable(tf.truncated_normal([N_HL9_UNITS, N_HL10_UNITS], stddev=1.0 / math.sqrt(float(N_HL9_UNITS))), name='weights_10')
    b_10 = tf.Variable(tf.ones([N_HL10_UNITS]), name='biases_10')
    l_10 = tf.nn.relu(tf.matmul(l_9, w_10) + b_10)

    # Output
    Y = l_10

    # Define loss and optimizer
    Y_ = tf.placeholder(tf.float32, [None, N_TARGET_NUM])

    mse = tf.losses.mean_squared_error(labels=Y_, predictions=Y)
    cost = mse
    train_step = tf.train.AdamOptimizer().minimize(cost)

    sess = tf.InteractiveSession()
    tf.global_variables_initializer().run()

    tf_epoch = tf.constant('epoch: ' + str(epoch))

    print(sess.run(tf_epoch))
    print('=======================================')

    # Train
    for _ in range(epoch):
        sess.run(train_step, feed_dict={X: train.data, Y_: train.target})

        if _ % 200 == 0:
            tmp_cost = sess.run(cost, feed_dict={X: train.data, Y_: train.target})
            print('')
            print('Iteration: ', _)
            print('Current cost on train dataset: ', tmp_cost)
            print('=======================================')


    # Test
    correct_prediction = tf.equal(tf.argmax(Y, 1), tf.argmax(Y_, 1))
    l = tf.scalar_mul(10, Y_)
    p = tf.round(tf.scalar_mul(10, Y))
    e = tf.reduce_mean(tf.cast(tf.equal(l, p), tf.float32))

    print("MSE: ", sess.run(cost, feed_dict={X: test.data, Y_: test.target}))
    print("Match Rate: ", sess.run(e, feed_dict={X: test.data, Y_: test.target}))

if __name__ == '__main__':
    tf.app.run(main=main)
