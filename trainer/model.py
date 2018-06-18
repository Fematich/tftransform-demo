#!/usr/bin/python
import tensorflow as tf


def inference(features):
    """Creates the predictions of the model

        Args:
          features (dict): A dictionary of tensors keyed by the feature name.

        Returns:
            A dict of tensors that represents the predictions

    """
    with tf.variable_scope('model'):
        input_features = tf.concat([tf.expand_dims(features[key],1) for key in
                                    [u'MixingTime', u'SugarHumidity', u'TotalMass', u'ButterTemperature',
                                     u'NormButterperc', u'ButterMass', u'HeatingTime',
                                     u'NormSugarperc', u'FlourHumidity']], 1)
        hidden = tf.layers.dense(inputs=input_features,
                                      units=5,
                                      name='dense_weights_1',
                                      use_bias=True)
        predictions = tf.layers.dense(inputs=hidden,
                                      units=1,
                                      name='dense_weights_2',
                                      use_bias=True)
        predictions_squeezed = tf.squeeze(predictions)
    return {'TotalVolume': predictions_squeezed}


def loss(predictions, labels):
    """Function that calculates the loss based on the predictions and labels

        Args:
          predictions (dict): A dictionary of tensors representing the predictions
          labels (dict): A dictionary of tensors representing the labels.

        Returns:
            A tensor representing the loss

    """
    with tf.variable_scope('loss'):
        return tf.losses.mean_squared_error(predictions['TotalVolume'], labels['TotalVolume'])


def build_model_fn():
    """Build model function as input for estimator.

    Returns:
        function: model function

    """

    def _model_fn(features, labels, mode, params):
        """Creates the prediction and its loss.

        Args:
          features (dict): A dictionary of tensors keyed by the feature name.
          labels (dict): A dictionary of tensors representing the labels.
          mode: The execution mode, defined in tf.estimator.ModeKeys.

        Returns:
          tf.estimator.EstimatorSpec: EstimatorSpec object containing mode,
          predictions, loss, train_op and export_outputs.

        """
        predictions = inference(features)
        loss_op = None
        train_op = None

        if mode != tf.estimator.ModeKeys.PREDICT:
            loss_op = loss(predictions, labels)

        if mode == tf.estimator.ModeKeys.TRAIN:
            train_op = tf.contrib.layers.optimize_loss(
                loss=loss_op,
                global_step=tf.train.get_global_step(),
                learning_rate=params['learning_rate'],
                optimizer='Adagrad',
                summaries=[
                    'learning_rate',
                    'loss',
                    'gradients',
                    'gradient_norm',
                ],
                name='train')

        export_outputs = {
            tf.saved_model.signature_constants.DEFAULT_SERVING_SIGNATURE_DEF_KEY:
                tf.estimator.export.PredictOutput(predictions)}

        return tf.estimator.EstimatorSpec(
            mode=mode,
            predictions=predictions,
            loss=loss_op,
            train_op=train_op,
            export_outputs=export_outputs)

    return _model_fn
