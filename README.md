# Object detection in urban environments

## Table of Contents
1. [Project Description](https://github.com/mose247/Object_detection_in_urban_environments/blob/main/README.md#project-description)
2. [Install & Run](https://github.com/mose247/Object_detection_in_urban_environments/blob/main/README.md#install--run)
3. [Methodology](https://github.com/mose247/Object_detection_in_urban_environments/blob/main/README.md#methodology)
4. [Future Work & Imporvements](https://github.com/mose247/Object_detection_in_urban_environments/blob/main/README.md#future-work--improvements)
## Project Description
In this project, pre-trained models from the [TensorFlow object detection API](https://tensorflow-object-detection-api-tutorial.readthedocs.io/en/latest/index.html) are fine-tuned in order to detect and classify cars, pedestrians and cyclists on the [Waymo Open Dataset](https://waymo.com/open/). In particular, experiments were conducted on **SSD MobileNet V2 FPNLite 640x640** and **SSD ResNet50 V1 FPN 640x640**, but you can find other models [here](https://github.com/tensorflow/models/blob/master/research/object_detection/g3doc/tf2_detection_zoo.md). 

The project leverages the following Amazon Web Services (AWS):
* [AWS Sagemaker](https://aws.amazon.com/sagemaker/) for training and deploying machine learning models;
* [AWS ECR](https://aws.amazon.com/ecr/?nc2=h_ql_prod_ct_ec2reg) for storing a docker container with all the dependencies required by the TensorFlow Object Detection API;
* [AWS S3](https://aws.amazon.com/s3/?nc2=h_ql_prod_st_s3) for storing tensorboard logs and accessing the dataset;
  
The dataset has already been exported using the [TFRecords format](https://tensorflow-object-detection-api-tutorial.readthedocs.io/en/latest/training.html#create-tensorflow-records) and stored in the public AWS S3 bucket `s3://cd2688-object-detection-tf2`. The images are saved with 640x640 resolution.

## Install & Run
To setup the project create a new SageMaker notebook instance and clone in it the Udacity's Object Detection in an Urban Environment [repository](https://github.com/udacity/cd2688-object-detection-in-urban-environment-project). Then, create a new S3 bucket for storing your personal tensorboard logs. 

> Note: while creating the notebook, make sure to attach the `AmazonS3FullAccess` and `AmazonEC2ContainerRegistryFullAccess` policies to its IAM Role in order to give your Sagemaker notebook instance access to S3 and ECR services.

Subsequently, add the files in the present repo to the cloned one by copying `my_train_models.ipynb` into the `1_model_training` directory and `my_deploy_models.ipynb` into the `2_model_inference` directory. This step ensure that you will have the relevant notebooks in their appropriate locations. 

Next, it is necessary to set up the configuration files to support the model training. To do this, add the provided config files to the `1_model_training/source_dir` directory. These configuration files play a crucial role in guiding the training process, so it's essential to have them in the right place.

Finally, to run the project and visualize the results:
1. run `my_train_models.ipynb` and follow the instructions in the notebook;
2. run `my_deploy_models.ipynb` and follow the instructions in the notebook;

## Methodology
For the purposes of this project no network will be train from scratch, but rather we will reuse the pre-trained **SSD MobileNet V2 FPNLite 640x640** and **SSD ResNet50 V1 FPN 640x640** provided by TensorFlow. In order to set up a new transfer learning job, the protobuf [files](https://github.com/tensorflow/models/tree/master/research/object_detection/configs/tf2) that were use to configure the training of the two networks on the [COCO 2017 dataset](https://cocodataset.org/#home) should be modified. In particular, to tune the models on the Waymo Open Dataset loaded in the AWS S3 public bucket, the following tweaks are necessary:

* Change the dimension of the output layer in the `model` field:
  *  `num_classes: 3`
* Set the path to the new training dataset and label map in the `train_input_reader` field:
  * `input_path: "/opt/ml/input/data/train/*.tfrecord"`
  * `label_map_path: "/opt/ml/input/data/train/label_map.pbtxt"`
* Set the path to the new evaluation dataset and label map in the `eval_input_reader` field:
  * `input_path: "/opt/ml/input/data/val/*.tfrecord"`
  * `label_map_path: "/opt/ml/input/data/train/label_map.pbtxt"`
* Set the path to the pre-trained weigths and specify the tuning type in the `train_config` field::
  *`fine_tune_checkpoint: "checkpoint/ckpt-0"`
  * `fine_tune_checkpoint_type: "detection"`


The **SSD MobileNet V2 FPNLite 640x640** and **SSD ResNet50 V1 FPN 640x640** models available in the TF2 Object Detection model zoo are trained on the [COCO 2017 dataset](https://cocodataset.org/#home), which is a large-scale object detection and segmentation dataset containing annotations for 90 different object classes.

### Modify output layer


### Add data augmentation strategies


### Experiment with annealing


### Results


## Future work & Improvements
