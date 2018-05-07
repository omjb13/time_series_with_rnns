import os
import shutil
import sys

import tensorflow as tf
from tensorflow.python.layers.core import Dense

import parse_input_parameters
from utils import input_fn, smape_fn

tf.logging.set_verbosity(tf.logging.INFO)


def model_fn(features, labels, mode, params):
    seq_inputs, nseq_inputs = features['seq'], features['nseq']
    seq_inputs = tf.identity(seq_inputs, name='seq_inputs')

    n_inputs_per_timestep = params['n_inputs_per_timestep']
    n_timesteps = params['n_timesteps']
    n_neurons = params['n_neurons']
    n_hidden = params['n_hidden']
    n_outputs = params['n_outputs']
    cell_type = params['cell_type']

    labels = tf.identity(labels, name='labels')

    batch_size = params['batch_size']

    seq = tf.reshape(seq_inputs, [-1, n_timesteps, n_inputs_per_timestep])

    # concatenate sequential and non-sequential features
    seq_time_major = tf.transpose(seq, perm=[1, 0, 2])
    splits = tf.split(seq_time_major, n_timesteps, axis=0)
    new_tensors = []
    for ts in splits:
        ts = tf.reshape(ts, [batch_size, 1])
        ts = tf.concat([ts, nseq_inputs], axis=1)
        ts = tf.expand_dims(ts, [-1])
        new_tensors.append(ts)
    final = tf.concat(new_tensors, axis=2)
    mixed_inputs = tf.transpose(final, perm=[2, 0, 1], name='mixed_inputs')

    labels = tf.reshape(labels, [n_outputs, -1])
    labels = tf.expand_dims(labels, -1)

    if cell_type == "fused_lstm":
        fused_rnn = tf.contrib.rnn.LSTMBlockFusedCell(num_units=n_neurons)
        outputs, encoder_state = fused_rnn(mixed_inputs, dtype=tf.float32)

    else:
        if cell_type == "gru":
            basic_cell = tf.nn.rnn_cell.GRUCell(num_units=n_neurons)

        elif cell_type == "lstm":
            basic_cell = tf.nn.rnn_cell.LSTMCell(num_units=n_neurons)

        else:
            basic_cell = tf.nn.rnn_cell.RNNCell(num_units=n_neurons)

        outputs, encoder_state = tf.nn.dynamic_rnn(
            basic_cell, mixed_inputs, dtype=tf.float32, time_major=True
        )

    # Build RNN cell
    decoder_cell = tf.nn.rnn_cell.LSTMCell(num_units=n_neurons)

    # Helper
    helper = tf.contrib.seq2seq.TrainingHelper(
        labels, [n_outputs for _ in range(batch_size)],
        time_major=True)

    projection_layer = Dense(units=1, use_bias=False)

    # Decoder
    decoder = tf.contrib.seq2seq.BasicDecoder(decoder_cell, helper,
                                              encoder_state,
                                              output_layer=projection_layer)

    # Dynamic decoding
    outputs, _, _ = tf.contrib.seq2seq.dynamic_decode(decoder,
                                                      output_time_major=True)
    predictions = outputs.rnn_output
    predictions = tf.identity(predictions, name='predictions')

    loss = tf.losses.mean_squared_error(labels, predictions)
    optimizer = tf.train.AdamOptimizer()
    gvs = optimizer.compute_gradients(loss)
    capped_gvs = [(tf.clip_by_value(grad, -1., 1.), var) for grad, var in gvs]
    optimize_op = optimizer.apply_gradients(capped_gvs,
                                            global_step=tf.train.get_global_step())

    if mode == tf.estimator.ModeKeys.TRAIN:
        return tf.estimator.EstimatorSpec(mode, loss=loss, train_op=optimize_op)

    elif mode == tf.estimator.ModeKeys.EVAL:
        return tf.estimator.EstimatorSpec(mode, loss=loss, eval_metric_ops={
            "smape": smape_fn(predictions, labels)
        })

    elif mode == tf.estimator.ModeKeys.PREDICT:
        return tf.estimator.EstimatorSpec(mode,
                                          predictions={'predicted': predictions})


def estimate(outputdir, datafile, params):
    train_spec = tf.estimator.TrainSpec(
        input_fn=lambda: input_fn(datafile, params, tf.estimator.ModeKeys.TRAIN),
        max_steps=params['n_steps'],
    )

    test_spec = tf.estimator.EvalSpec(
        input_fn=lambda: input_fn(datafile, params, tf.estimator.ModeKeys.EVAL),
        throttle_secs=60 * 10,
        start_delay_secs=60 * 20,
        steps=params["eval_steps"]
    )

    estimator = tf.estimator.Estimator(
        model_fn=model_fn,
        params=params,
        model_dir=outputdir,
    )

    tf.estimator.train_and_evaluate(estimator, train_spec, test_spec)


if __name__ == "__main__":
    params = parse_input_parameters.get_params(sys.argv[1:])
    output_path = params.pop("job_dir")

    if os.path.exists(output_path):
        shutil.rmtree(output_path, ignore_errors=True)

    hp_string = "neurons_{0}-hidden_{1}".format(params['n_neurons'],
                                                params['n_hidden'])

    print "Running job for : ", hp_string

    output_dir = output_path + "/" + hp_string
    datafile = params.pop("train_files")

    estimate(output_dir, datafile, params)
