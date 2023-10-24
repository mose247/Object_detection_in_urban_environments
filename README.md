# Object detection in urban environments
Object detection in urban environments

## Table of Contents

## Project Description
In this project, pre-trained models from the [TensorFlow object detection API](https://tensorflow-object-detection-api-tutorial.readthedocs.io/en/latest/index.html) are fine-tuned in order to detect and classify cars, pedestrians and cyclists on the [Waymo Open Dataset](https://waymo.com/open/). The project leverages the Amazon Web Services to train and deploy the models, specifically:
- [AWS Sagemaker](https://aws.amazon.com/sagemaker/) to train and deploy the models;
- [AWS ECR](https://aws.amazon.com/ecr/?nc2=h_ql_prod_ct_ec2reg) to store a docker container with all the dependencies required by the TF Object Detection API;
- [AWS S3](https://aws.amazon.com/s3/?nc2=h_ql_prod_st_s3) to store tensorboard logs;

## Installation

## Methodology

## Future work & Improvements
