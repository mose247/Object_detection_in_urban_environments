# Object detection in urban environments
Object detection in urban environments

## Table of Contents

## Project Description
In this project, pre-trained models from the [TensorFlow object detection API](https://tensorflow-object-detection-api-tutorial.readthedocs.io/en/latest/index.html) are fine-tuned in order to detect and classify cars, pedestrians and cyclists on the [Waymo Open Dataset](https://waymo.com/open/). In particular, experiments were conducted on **SSD MobileNet V2 FPNLite 640x640** and **SSD ResNet50 V1 FPN 640x640**, but you can find other models [here](https://github.com/tensorflow/models/blob/master/research/object_detection/g3doc/tf2_detection_zoo.md). 

The project leverages the following Amazon Web Services (AWS):
- [AWS Sagemaker](https://aws.amazon.com/sagemaker/) for training and deploying machine learning models;
- [AWS ECR](https://aws.amazon.com/ecr/?nc2=h_ql_prod_ct_ec2reg) for storing a docker container with all the dependencies required by the TF Object Detection API;
- [AWS S3](https://aws.amazon.com/s3/?nc2=h_ql_prod_st_s3) for storing tensorboard logs and accessing the dataset;
  
The dataset has already been exported using the [TFRecords format](https://tensorflow-object-detection-api-tutorial.readthedocs.io/en/latest/training.html#create-tensorflow-records) and stored in the public AWS S3 bucket `s3://cd2688-object-detection-tf2`. The images are saved with 640x640 resolution.

## Install & Run
To setup the project create a new SageMaker notebook and clone in it the [Udacity's Object Detection in an Urban Environment repository](https://github.com/udacity/cd2688-object-detection-in-urban-environment-project). Then, create a new S3 bucket for storing your personal tensorboard logs. 
> Note: while creating the notebook, make sure to attach the `AmazonS3FullAccess` and `AmazonEC2ContainerRegistryFullAccess` policies to its IAM Role in order to give your Sagemaker notebook access to S3 and ECR services.

## Methodology

## Future work & Improvements
