import os
import shutil
import sys

import tensorflow as tf

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
    cell_type = "rnn"

    labels = tf.identity(labels, name='labels')

    seq_inputs = tf.reshape(seq_inputs, [-1, n_timesteps, n_inputs_per_timestep])

    if cell_type == "fused_lstm":
        fused_rnn = tf.contrib.rnn.LSTMBlockFusedCell(num_units=n_neurons)
        outputs, states = fused_rnn(seq_inputs, dtype=tf.float32)
    else:
        if cell_type == "gru":
            basic_cell = tf.contrib.rnn.GRUCell(num_units=n_neurons)
        elif cell_type == "lstm":
            basic_cell = tf.contrib.rnn.BasicLSTMCell(num_units=n_neurons)
        else:
            # Use basic rnn by default
            basic_cell = tf.contrib.rnn.BasicRNNCell(num_units=n_neurons, name='rnn_cell')

        outputs, states = tf.nn.dynamic_rnn(
            basic_cell, seq_inputs, dtype=tf.float32,
        )

    # keep only the last timestep
    last_output = outputs[:, -1, :]

    hidden_activations = tf.contrib.layers.fully_connected(last_output, n_hidden, scope='fc_hidden')

    predictions = tf.contrib.layers.fully_connected(hidden_activations,
                                                    n_outputs,
                                                    activation_fn=None,
                                                    scope='fc_projection')

    tf.summary.histogram('rnn_states', states)
    tf.summary.histogram('output from RNN', outputs)
    trainables = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES)
    to_log = ['rnn_cell', 'fc_hidden', 'fc_projection']
    for variable in trainables:
        for target in to_log:
            if target in variable.name:
                print "adding histogram for %s" % variable.name
                tf.summary.histogram(variable.name, variable)

    predictions = tf.identity(predictions, name='predictions')

    loss = tf.losses.mean_squared_error(labels, predictions)

    optimizer = tf.train.AdamOptimizer(learning_rate=params['learning_rate'])
    gvs = optimizer.compute_gradients(loss)
    capped_gvs = [(tf.clip_by_value(grad, -1., 1.), var) for grad, var in gvs]
    optimize_op = optimizer.apply_gradients(capped_gvs, global_step=tf.train.get_global_step())

    if mode == tf.estimator.ModeKeys.TRAIN:
        return tf.estimator.EstimatorSpec(mode, loss=loss, train_op=optimize_op)

    elif mode == tf.estimator.ModeKeys.EVAL:
        return tf.estimator.EstimatorSpec(mode, loss=loss,
                                          eval_metric_ops={
                                              'smape': smape_fn(predictions, labels)
                                          })

    elif mode == tf.estimator.ModeKeys.PREDICT:
        return tf.estimator.EstimatorSpec(mode, predictions={'predicted': predictions})


def estimate(outputdir, datafile, params):
    train_spec = tf.estimator.TrainSpec(
        input_fn=lambda: input_fn(datafile, params, tf.estimator.ModeKeys.TRAIN),
        max_steps=params['n_steps'],
    )

    test_spec = tf.estimator.EvalSpec(
        input_fn=lambda: input_fn(datafile, params, tf.estimator.ModeKeys.EVAL),
        throttle_secs=(60 * 10),
        start_delay_secs=(60 * 20),
        steps=params['eval_steps']
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

    datafile = params.pop("train_files")

    estimate(output_path, datafile, params)
