#!/usr/bin/python
import logging
import tensorflow as tf

from trainer.util import build_training_input_fn, build_serving_input_fn
from trainer.model import build_model_fn
from trainer.config import MODEL_DIR

if __name__ == '__main__':
    logger = logging.getLogger('task')
    logging.basicConfig(format='%(asctime)s %(message)s')
    logger.setLevel('INFO')
    estimator = tf.estimator.Estimator(
        model_fn=build_model_fn(),
        model_dir=MODEL_DIR,
        params={'learning_rate': 0.001})

    train_spec = tf.estimator.TrainSpec(input_fn=build_training_input_fn(), max_steps=200)
    eval_spec = tf.estimator.EvalSpec(input_fn=build_training_input_fn(), steps=64)
    tf.estimator.train_and_evaluate(estimator, train_spec, eval_spec)
    logger.info('model is trained')

    serving_input_fn = build_serving_input_fn()
    estimator.export_savedmodel(
        MODEL_DIR, serving_input_fn)
    logger.info('model is saved')
