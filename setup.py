#!/usr/bin/python
from setuptools import find_packages
from setuptools import setup

setup(
    name='tft-demo',
    version='0.2',
    author='Matthias Feys',
    author_email='matthiasfeys@gmail.com',
    install_requires=['tensorflow==1.8.0',
                      'tensorflow-transform==0.8.0'],
    packages=find_packages(exclude=['data']),
    description='tf.Transform demo for digital twin',
    url='https://github.com/Fematich/tftransform-demo/'
)
