#-*- coding: utf-8 -*-

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf

from datasets.pension_lottery import datasets

N_INPUT_DATA = 100

INPUT_SIZE    = 43
RNN_HIDDEN    = 43
OUTPUT_SIZE   = 35

epoch = 500

USE_LSTM = True

def main(_):
    data = datasets.load_pension_lottery_rnn_datasets(n_data=N_INPUT_DATA, shuffle=True, ratio=[0.7, None, 0.3])

    train = data.train
    test = data.test

    X  = tf.placeholder(tf.float32, (None, None, INPUT_SIZE))  # (time, batch, in)

    if USE_LSTM:
        cell = tf.nn.rnn_cell.BasicLSTMCell(RNN_HIDDEN, state_is_tuple=True)
    else:
        cell = tf.nn.rnn_cell.BasicRNNCell(RNN_HIDDEN)

    batch_size    = tf.shape(X)[1]
    initial_state = cell.zero_state(batch_size, tf.float32)

    rnn_outputs, rnn_states = tf.nn.dynamic_rnn(cell, X, initial_state=initial_state, time_major=True)

    final_projection = lambda x: tf.contrib.layers.linear(x, num_outputs=OUTPUT_SIZE, activation_fn=tf.nn.relu)

    Y = tf.map_fn(final_projection, rnn_outputs)

    Y_ = tf.placeholder(tf.float32, (None, None, OUTPUT_SIZE)) # (time, batch, out)

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
    l = tf.scalar_mul(10, Y_)
    p = tf.round(tf.scalar_mul(10, Y))
    e = tf.reduce_mean(tf.cast(tf.equal(l, p), tf.float32))

    print("MSE: ", sess.run(cost, feed_dict={X: test.data, Y_: test.target}))
    print("Match Rate: ", sess.run(e, feed_dict={X: test.data, Y_: test.target}))

if __name__ == '__main__':
    tf.app.run(main=main)
