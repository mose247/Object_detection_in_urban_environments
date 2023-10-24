# Object detection in urban environments
Object detection in urban environments

## Table of Contents

## Project Description
In this project, pre-trained models from the [TensorFlow object detection API](https://tensorflow-object-detection-api-tutorial.readthedocs.io/en/latest/index.html) are fine-tuned in order to detect and classify cars, pedestrians and cyclists on the [Waymo Open Dataset](https://waymo.com/open/). The project leverages the following Amazon Web Services (AWS):

- [AWS Sagemaker](https://aws.amazon.com/sagemaker/) for training and deploying machine learning models;
- [AWS ECR](https://aws.amazon.com/ecr/?nc2=h_ql_prod_ct_ec2reg) for storing a docker container with all the dependencies required by the TF Object Detection API;
- [AWS S3](https://aws.amazon.com/s3/?nc2=h_ql_prod_st_s3) for storing tensorboard logs and accessing the dataset;
  
The dataset has already been exported using the [TFRecords format](https://tensorflow-object-detection-api-tutorial.readthedocs.io/en/latest/training.html#create-tensorflow-records) and stored in the public AWS S3 bucket `s3://cd2688-object-detection-tf2`. The images are saved with 640x640 resolution.

## Install & Run
To setup the project create a new SageMaker notebook instance and clone in it the [Udacity's Object Detection in an Urban Environment repository](https://github.com/udacity/cd2688-object-detection-in-urban-environment-project).

## Methodology

## Future work & Improvements
