#!/usr/bin/python
import argparse
import logging
import os
import sys
import tempfile
from datetime import datetime

import apache_beam as beam
import tensorflow as tf
import tensorflow_transform as tft
from apache_beam.io import tfrecordio
from tensorflow_transform import coders
from tensorflow_transform.beam import impl as beam_impl
from tensorflow_transform.beam.tft_beam_io import transform_fn_io
from tensorflow_transform.tf_metadata import dataset_metadata

from trainer.config import BUCKET, TRAIN_INPUT_DATA, TRAIN_OUTPUT_DATA, TFRECORD_DIR, MODEL_DIR, \
    input_schema, output_schema, example_schema

delimiter = ';'
converter_input = coders.CsvCoder(
    ['BatchId', 'ButterMass', 'ButterTemperature', 'SugarMass', 'SugarHumidity', 'FlourMass', 'FlourHumidity',
     'HeatingTime', 'MixingSpeed', 'MixingTime'],
    input_schema,
    delimiter=delimiter)
converter_output = coders.CsvCoder(
    ['BatchId', 'TotalVolume', 'Density', 'Temperature', 'Humidity', 'Energy', 'Problems'],
    output_schema,
    delimiter=delimiter)
input_metadata = dataset_metadata.DatasetMetadata(schema=example_schema)


def extract_batchkey(record):
    """Extracts the BatchId out of the record
        Args:
            record (dict): record of decoded CSV line
        Returns:
            tuple: tuple of BatchId and record
    """
    return (record['BatchId'], record)


def remove_keys(item):
    """Clean CoGroupByKey result by removing the keys
    Args:
        item: result of CoGroupByKey
    Returns:
        dict: dict with item removed of the key
    """
    key, vals = item
    if len(vals[0]) == 1 and len(vals[1]) == 1:
        example = vals[0][0]
        example.update(vals[1][0])
        yield example


def preprocessing_fn(inputs):
    """
    Preprocess input columns into transformed columns.
    Args:
        inputs (dict): dict of input columns
    Returns:
        output dict of transformed columns
    """
    outputs = {}
    # Encode categorical column:
    outputs['MixingSpeed'] = tft.string_to_int(inputs['MixingSpeed'])
    outputs['ButterMass'] = inputs['ButterMass']
    # Calculate Derived Features:
    outputs['TotalMass'] = inputs['ButterMass'] + inputs['SugarMass'] + inputs['FlourMass']
    for ingredient in ['Butter', 'Sugar', 'Flour']:
        ingredient_percentage = inputs['{}Mass'.format(ingredient)] / outputs['TotalMass']
        outputs['Norm{}perc'.format(ingredient)] = tft.scale_to_z_score(ingredient_percentage)
    # Keep absolute numeric columns
    for key in ['TotalVolume', 'Energy']:
        outputs[key] = inputs[key]
    # Normalize other numeric columns
    for key in [
        'ButterTemperature',
        'SugarHumidity',
        'FlourHumidity',
        'HeatingTime',
        'MixingTime',
        'Density',
        'Temperature',
        'Humidity',
    ]:
        outputs[key] = tft.scale_to_z_score(inputs[key])
    # Extract Specific Problems
    chunks_detected_str = tf.regex_replace(
        input=inputs['Problems'],
        pattern='.*chunk.*',
        rewrite='chunk',
        name='DetectChunk')
    outputs['Chunks'] = tf.cast(tf.equal(chunks_detected_str, 'chunk'), tf.float32)
    return outputs


def parse_arguments(argv):
    """Parse command line arguments
    Args:
        argv (list): list of command line arguments including program name
    Returns:
        The parsed arguments as returned by argparse.ArgumentParser
    """
    parser = argparse.ArgumentParser(description='Runs Preprocessing.')
    parser.add_argument('--cloud',
                        action='store_true',
                        help='Run preprocessing on the cloud.')
    args, _ = parser.parse_known_args(args=argv[1:])
    return args


def get_cloud_pipeline_options(project, output_dir):
    """Get apache beam pipeline options to run with Dataflow on the cloud
    Args:
        project (str): GCP project to which job will be submitted
        output_dir (str): GCS directory to which output will be written
    Returns:
        beam.pipeline.PipelineOptions
    """
    logging.warning('Start running in the cloud')

    options = {
        'runner': 'DataflowRunner',
        'job_name': ('preprocessdigitaltwin-{}'.format(
            datetime.now().strftime('%Y%m%d%H%M%S'))),
        'staging_location': os.path.join(BUCKET, 'staging'),
        'temp_location': os.path.join(BUCKET, 'tmp'),
        'project': project,
        'region': 'europe-west1',
        'zone': 'europe-west1-d',
        'autoscaling_algorithm': 'THROUGHPUT_BASED',
        'save_main_session': True,
        'setup_file': './setup.py',
    }

    return beam.pipeline.PipelineOptions(flags=[], **options)


def main(argv=None):
    """Run preprocessing as a Dataflow pipeline.
    Args:
        argv (list): list of arguments
    """
    args = parse_arguments(sys.argv if argv is None else argv)

    if args.cloud:
        pipeline_options = get_cloud_pipeline_options(args.project_id,
                                                      args.output_dir)
    else:
        pipeline_options = None

    p = beam.Pipeline(options=pipeline_options)
    with beam_impl.Context(temp_dir=tempfile.mkdtemp()):
        # read data and join by key
        raw_data_input = (
            p
            | 'ReadInputData' >> beam.io.ReadFromText(TRAIN_INPUT_DATA, skip_header_lines=1)
            | 'ParseInputCSV' >> beam.Map(converter_input.decode)
            | 'ExtractBatchKeyIn' >> beam.Map(extract_batchkey)
        )

        raw_data_output = (
            p
            | 'ReadOutputData' >> beam.io.ReadFromText(TRAIN_OUTPUT_DATA, skip_header_lines=1)
            | 'ParseOutputCSV' >> beam.Map(converter_output.decode)
            | 'ExtractBatchKeyOut' >> beam.Map(extract_batchkey)
        )

        raw_data = (
            (raw_data_input, raw_data_output)
            | 'JoinData' >> beam.CoGroupByKey()
            | 'RemoveKeys' >> beam.FlatMap(remove_keys)
        )

        # analyse and transform dataset
        raw_dataset = (raw_data, input_metadata)
        transform_fn = raw_dataset | beam_impl.AnalyzeDataset(preprocessing_fn)
        transformed_dataset = (raw_dataset, transform_fn) | beam_impl.TransformDataset()
        transformed_data, transformed_metadata = transformed_dataset

        # save data and serialize TransformFn
        transformed_data_coder = tft.coders.ExampleProtoCoder(
            transformed_metadata.schema)
        _ = (transformed_data
             | 'EncodeData' >> beam.Map(transformed_data_coder.encode)
             | 'WriteData' >> tfrecordio.WriteToTFRecord(
            os.path.join(TFRECORD_DIR, 'records')))
        _ = (transform_fn
             | "WriteTransformFn" >>
             transform_fn_io.WriteTransformFn(MODEL_DIR))

        p.run().wait_until_finish()


if __name__ == '__main__':
    main()
