#!/usr/bin/python
"""
Part of the demo responsible for training and exporting the model
"""
import logging
import tensorflow as tf
import tensorflow_transform as tft

from trainer.util import build_training_input_fn, build_serving_input_fn
from trainer.model import build_model_fn
from trainer.config import MODEL_DIR

if __name__ == '__main__':
    LOGGER = logging.getLogger('task')
    logging.basicConfig(format='%(asctime)s %(message)s')
    LOGGER.setLevel('INFO')
    ESTIMATOR = tf.estimator.Estimator(
        model_fn=build_model_fn(),
        model_dir=MODEL_DIR,
        params={'learning_rate': 0.001})

    TRAIN_SPEC = tf.estimator.TrainSpec(input_fn=build_training_input_fn(), max_steps=200)
    EVAL_SPEC = tf.estimator.EvalSpec(input_fn=build_training_input_fn(), steps=64)
    tf.estimator.train_and_evaluate(ESTIMATOR, TRAIN_SPEC, EVAL_SPEC)
    LOGGER.info('model is trained')

    TF_TRANSFORM_OUTPUT = tft.TFTransformOutput(MODEL_DIR)
    SERVING_INPUT_FN = build_serving_input_fn(TF_TRANSFORM_OUTPUT)
    ESTIMATOR.export_savedmodel(
        MODEL_DIR, SERVING_INPUT_FN)
    LOGGER.info('model is saved')
