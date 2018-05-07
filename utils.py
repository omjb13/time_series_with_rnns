import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.python.framework import ops
from tensorflow.python.lib.io import file_io


def get_max(df):
    cols_to_convert = ['name', 'country_code', 'type', 'access']
    translation_dicts = {}
    for i, col in enumerate(cols_to_convert):
        this_col = df[col]
        translation_dicts[i + 2] = int(this_col.max()) + 1
    return translation_dicts


def one_hot_encode(features, labels, translation_dict):
    nseq = features['nseq']

    nseq_tensors = tf.split(nseq, 6, 0)
    to_encode = [3, 4, 5]
    encoded = tf.concat([nseq_tensors[0], nseq_tensors[1]], 0)

    for col in to_encode:
        one_hot = tf.one_hot(tf.cast(nseq_tensors[col], tf.int32),
                             translation_dict[col])
        one_hot = tf.reshape(one_hot, [-1])
        mean, var = tf.nn.moments(one_hot, [0], keep_dims=True)
        one_hot = tf.div(tf.subtract(one_hot, mean), tf.sqrt(var))
        encoded = tf.concat([encoded, one_hot], 0)

    features['nseq'] = encoded
    return features, labels


def input_fn(dataset, config, mode=tf.estimator.ModeKeys.TRAIN, method=1):
    with tf.name_scope('data_input'):
        if dataset.startswith("gs://"):
            with file_io.FileIO(dataset, mode="r") as f:
                train = pd.read_csv(f, index_col=0).astype(np.float32)
        else:
            train = pd.read_csv(dataset, index_col=0).astype(np.float32)

        # split into sequential and non-sequential
        seq = train.iloc[:, :-6]
        nseq = train.iloc[:, -6:]
        # normalize sequential data
        seq = seq.apply(np.log1p)
        # get configs
        tsteps = config['n_timesteps']
        n_outputs = config['n_outputs']
        # build dataset
        if mode == tf.estimator.ModeKeys.TRAIN:
            # training features : 0 - 100
            seq_inputs = seq.iloc[:, 0:tsteps]
            if method == 1:
                # train labels 1: 100 - 110
                labels = seq.iloc[:, tsteps:tsteps + n_outputs]
            else:
                # train labels 2 : 1 - 101
                labels = seq.iloc[:, 1:tsteps + 1]

        elif tf.estimator.ModeKeys.EVAL:
            # test features : 10 - 110
            seq_inputs = seq.iloc[:, n_outputs: tsteps + n_outputs]
            if method == 1:
                # test labels 1 : 110 - 120
                labels = seq.iloc[:,
                         tsteps + n_outputs: tsteps + 2 * n_outputs]
            else:
                # test labels 2 : 11 - 111
                labels = seq.iloc[:, n_outputs + 1: tsteps + n_outputs + 1]

        features = {'nseq': nseq, 'seq': seq_inputs}

        translation_dict = get_max(nseq)

        dataset = tf.data.Dataset.from_tensor_slices((features, labels))
        dataset = dataset.map(
            lambda features, labels: one_hot_encode(features, labels, translation_dict)
        )

        if mode == tf.estimator.ModeKeys.TRAIN:
            dataset = dataset.shuffle(100000).repeat().batch(
                config['batch_size'])
        elif mode == tf.estimator.ModeKeys.EVAL:
            dataset = dataset.batch(config['batch_size'])
            # dataset = dataset.apply(tf.contrib.data.batch_and_drop_remainder(config['batch_size']))

        return dataset.make_one_shot_iterator().get_next()


def smape_fn(predictions, labels):
    y_pred = tf.expm1(tf.reshape(predictions, [-1]))
    y_true = tf.expm1(tf.reshape(labels, [-1]))

    diffs = tf.get_variable('diffs', [],
                            trainable=False,
                            collections=[ops.GraphKeys.LOCAL_VARIABLES,
                                         ops.GraphKeys.METRIC_VARIABLES])

    count = tf.get_variable('count', [],
                            trainable=False,
                            collections=[ops.GraphKeys.LOCAL_VARIABLES,
                                         ops.GraphKeys.METRIC_VARIABLES])

    with tf.control_dependencies([y_pred, y_true]):
        update_diffs = tf.assign_add(diffs, tf.reduce_sum(
            tf.abs(y_true - y_pred) / (tf.abs(y_true) + tf.abs(y_pred))
        ))
        update_count = tf.assign_add(count, tf.cast(tf.size(y_true), tf.float32))

    smape = (tf.constant(200.0) / count) * diffs
    update_smape = (tf.constant(200.0) / update_count) * update_diffs

    return smape, update_smape
