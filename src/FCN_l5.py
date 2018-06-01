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
N_HL5_UNITS = N_TARGET_NUM

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

    # Output
    Y = l_5

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


"""
#-*- coding: utf-8 -*-

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf

from datasets.pension_lottery import datasets

def FCN_multi_output_regression(features, labels, mode, params):
    top = tf.feature_column.input_layer(features, params['feature_columns'])

    for units in params.get('hidden_uints', [72]):
        top = tf.layers.dense(inputs=top, units=units, activation=tf.nn.relu)

    output_layer = tf.layers.dense(inputs=top, units=params['output_columns'])
    predictions = tf.squeeze(output_layer)

    if mode == tf.estimator.ModeKeys.PREDICT::
        return tf.estimator.EstimatorSpec(mode=mode, predictions={'prediction': predictions})

    average_loss = tf.losses.mean_squared_error(labels, predictions)

    batch_size = tf.shape(labels)[0]
    total_loss = tf.to_float(batch_size) * average_loss

    if mode == tf.estimator.ModeKeys.TRAIN:
        optimizer = params.get('optimizer', tf.train.AdamOptimizer)
        optimizer = optimizer(params.get('learning_rate', None))
        train_op = optimizer.minimize(loss=average_loss, global_step=tf.train.get_global_step()))

    return tf.estimator.EstimatorSpec(mode=mode, loss=total_loss, train_op=train_op)

    assert mode == tf.estimator.ModeKeys.EVAL

    rmse = tf.metrics.root_mean_squared_error(labels, predictions)

    eval_metrics = {"rmse": rmse}
    return tf.estimator.EstimatorSpec(mode=mode,loss=total_loss,eval_metric_ops=eval_metrics)

def main(argv):
    assert len(argv) == 1
    data = datasets.load_pension_lottery_fcn_datasets(n_data=10, shuffle=True, ratio=[0.7, None, 0.3])

    train = data.train
    test = data.test

  # The first way assigns a unique weight to each category. To do this you must
  # specify the category's vocabulary (values outside this specification will
  # receive a weight of zero). Here we specify the vocabulary using a list of
  # options. The vocabulary can also be specified with a vocabulary file (using
  # `categorical_column_with_vocabulary_file`). For features covering a
  # range of positive integers use `categorical_column_with_identity`.
  body_style_vocab = ["hardtop", "wagon", "sedan", "hatchback", "convertible"]
  body_style = tf.feature_column.categorical_column_with_vocabulary_list(
      key="body-style", vocabulary_list=body_style_vocab)
  make = tf.feature_column.categorical_column_with_hash_bucket(
      key="make", hash_bucket_size=50)

  feature_columns = [
      tf.feature_column.numeric_column(key="curb-weight"),
      tf.feature_column.numeric_column(key="highway-mpg"),
      # Since this is a DNN model, convert categorical columns from sparse
      # to dense.
      # Wrap them in an `indicator_column` to create a
      # one-hot vector from the input.
      tf.feature_column.indicator_column(body_style),
      # Or use an `embedding_column` to create a trainable vector for each
      # index.
      tf.feature_column.embedding_column(make, dimension=3),
  ]

  # Build a custom Estimator, using the model_fn.
  # `params` is passed through to the `model_fn`.
  model = tf.estimator.Estimator(
      model_fn=my_dnn_regression_fn,
      params={
          "feature_columns": feature_columns,
          "learning_rate": 0.001,
          "optimizer": tf.train.AdamOptimizer,
          "hidden_units": [20, 20]
      })

  # Train the model.
  model.train(input_fn=input_train, steps=STEPS)

  # Evaluate how the model performs on data it has not yet seen.
  eval_result = model.evaluate(input_fn=input_test)

  # Print the Root Mean Square Error (RMSE).
  print("\n" + 80 * "*")
  print("\nRMS error for the test set: ${:.0f}"
        .format(PRICE_NORM_FACTOR * eval_result["rmse"]))

  print()



if __name__ == '__main__':
    data = datasets.load_pension_lottery_fcn_datasets(n_data=10, shuffle=True, ratio=[0.7, None, 0.3])

    print(len(data.train.target[0]))
    print(len(data.train.data[0]))
"""
