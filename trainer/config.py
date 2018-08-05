#!/usr/bin/python
"""
Definition of all the variables
"""
import tensorflow as tf
from tensorflow_transform.tf_metadata import dataset_schema
from secrets import PROJECT_ID, BUCKET

DATA_DIR = BUCKET + '/data'
TRAIN_INPUT_DATA = DATA_DIR + '/input_data.csv'
TRAIN_OUTPUT_DATA = DATA_DIR + '/output_data.csv'
TFRECORD_DIR = BUCKET + '/tfrecords'
MODEL_DIR = BUCKET + '/model'
BATCH_SIZE = 64

INPUT_SCHEMA = dataset_schema.from_feature_spec({
    'BatchId': tf.FixedLenFeature(shape=[], dtype=tf.string),
    'ButterMass': tf.FixedLenFeature(shape=[], dtype=tf.float32),
    'ButterTemperature': tf.FixedLenFeature(shape=[], dtype=tf.float32),
    'SugarMass': tf.FixedLenFeature(shape=[], dtype=tf.float32),
    'SugarHumidity': tf.FixedLenFeature(shape=[], dtype=tf.float32),
    'FlourMass': tf.FixedLenFeature(shape=[], dtype=tf.float32),
    'FlourHumidity': tf.FixedLenFeature(shape=[], dtype=tf.float32),
    'HeatingTime': tf.FixedLenFeature(shape=[], dtype=tf.float32),
    'MixingSpeed': tf.FixedLenFeature(shape=[], dtype=tf.string),
    'MixingTime': tf.FixedLenFeature(shape=[], dtype=tf.float32),
})

OUTPUT_SCHEMA = dataset_schema.from_feature_spec({
    'BatchId': tf.FixedLenFeature(shape=[], dtype=tf.string),
    'TotalVolume': tf.FixedLenFeature(shape=[], dtype=tf.float32),
    'Density': tf.FixedLenFeature(shape=[], dtype=tf.float32),
    'Temperature': tf.FixedLenFeature(shape=[], dtype=tf.float32),
    'Humidity': tf.FixedLenFeature(shape=[], dtype=tf.float32),
    'Energy': tf.FixedLenFeature(shape=[], dtype=tf.float32),
    'Problems': tf.FixedLenFeature(shape=[], dtype=tf.string),
})

EXAMPLE_SCHEMA = dataset_schema.from_feature_spec({
    'ButterMass': tf.FixedLenFeature(shape=[], dtype=tf.float32),
    'ButterTemperature': tf.FixedLenFeature(shape=[], dtype=tf.float32),
    'SugarMass': tf.FixedLenFeature(shape=[], dtype=tf.float32),
    'SugarHumidity': tf.FixedLenFeature(shape=[], dtype=tf.float32),
    'FlourMass': tf.FixedLenFeature(shape=[], dtype=tf.float32),
    'FlourHumidity': tf.FixedLenFeature(shape=[], dtype=tf.float32),
    'HeatingTime': tf.FixedLenFeature(shape=[], dtype=tf.float32),
    'MixingSpeed': tf.FixedLenFeature(shape=[], dtype=tf.string),
    'MixingTime': tf.FixedLenFeature(shape=[], dtype=tf.float32),
    'TotalVolume': tf.FixedLenFeature(shape=[], dtype=tf.float32),
    'Density': tf.FixedLenFeature(shape=[], dtype=tf.float32),
    'Temperature': tf.FixedLenFeature(shape=[], dtype=tf.float32),
    'Humidity': tf.FixedLenFeature(shape=[], dtype=tf.float32),
    'Energy': tf.FixedLenFeature(shape=[], dtype=tf.float32),
    'Problems': tf.FixedLenFeature(shape=[], dtype=tf.string),
})
