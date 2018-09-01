tf.Transform example for building digital twin
====================

This repository is designed to quickly get you started with Machine Learning projects on Google Cloud Platform using tf.Transform.
This code repository is linked to [this Google Cloud blogpost](https://cloud.google.com/blog/products/ai-machine-learning/pre-processing-tensorflow-pipelines-tftransform-google-cloud) 

For more boilerplate examples, check: https://github.com/Fematich/mlengine-boilerplate

### Functionalities
- preprocessing pipeline using tf.Transform (with Apache Beam) that runs on Cloud Dataflow or locally
- model training (with Tensorflow) that runs locally or on ML Engine
- ready to deploy saved models to deploy on ML Engine
- starter code to use the saved model on ML Engine

### Install dependencies
**Note** You will need a Linux or Mac environment with Python 2.7.x to install the dependencies <sup>[1](#myfootnote1)</sup>.

Install the following dependencies:
 * Install [Cloud SDK](https://cloud.google.com/sdk/)
 * Install [gcloud](https://cloud.google.com/sdk/gcloud/)
 * ```pip install -r requirements.txt```

# Getting started

You need to complete the following parts to run the code:
- add trainer/secrets.py with your `PROJECT_ID` and `BUCKET` variable
- upload data to your buckets, you can upload data/test.csv to test this code

## Preprocess

You can run preprocess.py in the cloud using:
```
python preprocess.py --cloud
      
```

To iterate/test your code, you can also run it locally on a sample of the dataset:
```
python preprocess.py
```

## Training Tensorflow model
You can submit a ML Engine training job with:
```
gcloud ml-engine jobs submit training my_job \
                --module-name trainer.task \
                --staging-bucket gs://<staging_bucket> \
                --package-path trainer
```
Testing it locally:
```
gcloud ml-engine local train --package-path trainer \
                           --module-name trainer.task
```

## Deploy your trained model
To deploy your model to ML Engine
```
gcloud ml-engine models create digitaltwin
gcloud ml-engine versions create v1 --model=digitaltwin --origin=ORIGIN
```
To test the deployed model:
```
python predict.py
```


<a name="myfootnote1">1</a>: This code requires both Tensorflow and Apache Beam. Currently Tensorflow on Windows only supports Python 3.5.x and 
and Apache Beam doesn't support Python 3.x yet.
