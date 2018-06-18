#!/usr/bin/python
import os
import tensorflow as tf

from tensorflow_transform.beam.tft_beam_io import transform_fn_io
from tensorflow_transform.saved import saved_transform_io
from tensorflow_transform.tf_metadata import metadata_io

from trainer.config import TFRECORD_DIR, BATCH_SIZE, MODEL_DIR
from trainer.config import input_schema

def build_training_input_fn():
    """Creates an input function reading from transformed data.
    Args:
      transformed_examples: Base filename of examples.
    Returns:
      The input function for training or eval.
    """
    transformed_metadata = metadata_io.read_metadata(
        os.path.join(
            MODEL_DIR, transform_fn_io.TRANSFORMED_METADATA_DIR))
    transformed_feature_spec = transformed_metadata.schema.as_feature_spec()

    def input_fn():
        """Input function for training and eval."""
        dataset = tf.contrib.data.make_batched_features_dataset(
            file_pattern=TFRECORD_DIR,
            batch_size=BATCH_SIZE,
            features=transformed_feature_spec,
            reader=tf.data.TFRecordDataset,
            shuffle=True)
        transformed_features = dataset.make_one_shot_iterator().get_next()
        # Extract features and labels from the transformed tensors.
        label_cols = set(['TotalVolume', 'Density', 'Temperature', 'Humidity', 'Energy', 'Problems'])
        transformed_labels = {key: value for (key, value) in transformed_features.items() if key in label_cols}
        transformed_features = {key: value for (key, value) in transformed_features.items() if key not in label_cols}
        return transformed_features, transformed_labels

    return input_fn

def build_serving_input_fn():
  """Creates an input function reading from raw data.
  Args:
  Returns:
    The serving input function.
  """
  raw_feature_spec = input_schema.as_feature_spec()
  raw_feature_spec.pop('BatchId')

  def serving_input_fn():
    """Input function for serving."""
    # Get raw features by generating the basic serving input_fn and calling it.
    # Here we generate an input_fn that expects a parsed Example proto to be fed
    # to the model at serving time.  See also
    # tf.estimator.export.build_raw_serving_input_receiver_fn.
    raw_input_fn = tf.estimator.export.build_parsing_serving_input_receiver_fn(
        raw_feature_spec)
    raw_features, _, default_inputs = raw_input_fn()
    # Apply the transform function that was used to generate the materialized
    # data.
    _, transformed_features = (
        saved_transform_io.partially_apply_saved_transform(
            os.path.join(MODEL_DIR, transform_fn_io.TRANSFORM_FN_DIR),
            raw_features))

    return tf.estimator.export.ServingInputReceiver(transformed_features, raw_features)

  return serving_input_fn


