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

* Change the dimension of the output layer in `model` to match the new classification task. Since the Waymo Open Dataset contains annotations for three classes (car, pedestrian and cyclist), we should set:
  * `num_classes: 3`
    
* Set the path to the new training dataset and label map in `train_input_reader`:
  * `input_path: "/opt/ml/input/data/train/*.tfrecord"`
  * `label_map_path: "/opt/ml/input/data/train/label_map.pbtxt"`
     
* Set the path to the new evaluation dataset and label map in `eval_input_reader`:
  * `input_path: "/opt/ml/input/data/val/*.tfrecord"`
  * `label_map_path: "/opt/ml/input/data/train/label_map.pbtxt"`
    
* Set the path to the pre-trained weigths and specify the tuning type in `train_config`:
  * `fine_tune_checkpoint: "checkpoint/ckpt-0"`
  * `fine_tune_checkpoint_type: "detection"`

Additionaly, different strategies for optimization, learning rates, and data augmentations can be explored to enhance models performance. Due to limitations on the AWS budget, models are trained using just **2000 updating steps** and **batches** of **8** images. This helps completing trainings within reasonable time and avoiding capacity errors.

To minimize the loss function, **Stochastic Gradient Descent (SGD) with Momentum** is utilized. This optimization technique is known for its faster convergence and increased robustness to local minima compared to Vanilla SGD. Nevertheless, it tends to introduce more oscillations. To address this issue, a cosine annealing strategy with linear warm-up is adopted. The warm-up phase serves to mitigate potential early overfitting, while the decay phase is designed to reduce oscillations in the later stages of training.

```
  optimizer {
    momentum_optimizer: {
      learning_rate: {
        cosine_decay_learning_rate {
          learning_rate_base: .04         # learning rate to reach at the end of the warmup
          total_steps: 30000              # number of decay steps
          warmup_learning_rate: .001      # small initial learning rate
          warmup_steps: 500               # number of warmup steps
        }
      }
      momentum_optimizer_value: 0.9       #  momentum coefficient 
    }
    use_moving_average: false
  }
```

Finally, the following data augmentations are incorporated to diversify the training dataset and improve models' generalization capabilities.

* **Random Horizontal Flip**: flipping images horizontally expose models to variations of object orientations. This helps in making the model more robust to objects approaching from different directions and angles.
  ```
  data_augmentation_options {
    random_horizontal_flip {
    }
  }
  ```

* **Random Crop**: random cropping of the input images enables models to learn how to hanlde objects of different size and aspect ratio. This can be particularly important for accurately detecting objects that can appear at different distances.
  ```
  data_augmentation_options {
    random_crop_image {
      min_object_covered: 0.0
      min_aspect_ratio: 0.75
      max_aspect_ratio: 3.0
      min_area: 0.75
      max_area: 1.0
      overlap_thresh: 0.0
    }
  }
  ```
* **Random Adjust Brightness**: adjusting the brightness introduces variations in lighting conditions. This helps the model become more robust to different lighting scenarios, such as nighttime, daytime or overcast weather.
  ```
  data_augmentation_options {
    random_adjust_brightness {
      max_delta: 0.05
    }
  }
  ```
* **Random Black Patches**: adding random black patches to the input images mimics occluded objects in real-world. This improves models' ability to detect objects even when they are not entirely visible, which is common in urban traffic scenes.
  ```
  data_augmentation_options {
    random_black_patches {
      max_black_patches: 10
      size_to_image_ratio: 0.1
    }
  }
  ```

## Results
The graph below shows the total training loss (localization + classification + regularization) obtained for the MobileNet V2 (in blue) and the ResNet50 V1 (in orange) after 2000 updating steps. While ResNet50 V1 didn't converge, MobileNet V2 reached a plateau. Nevertheless, even in the second case, the total loss is still quite high, which may be a sign that the optimization got stuck in a local minimum.

<p align="center">
<img src="https://github.com/mose247/Object_detection_in_urban_environments/assets/91455159/3d1af224-f4d5-4bbe-a329-4fa923e757ce" title="Total loss" width=50% height=50%>
<em>image_caption</em>
</p>

Unfortunately, it wasn't possible to further evaluate the mAP of the models on the validation dataset due to some errors appeared during the validation phase. Nevertheless, the two videos before show a side-to-side comparison of the models on 100 frames.

<p align="center">
<figure>
  <img src="https://github.com/mose247/Object_detection_in_urban_environments/blob/main/data/output_resnet50.gif" title="ResNet50" width=40% height=40%>
  <figcaption> ResNet50 </figcaption>
</figure>
<figure>
  <img src="https://github.com/mose247/Object_detection_in_urban_environments/blob/main/data/output_mobilenet.gif" title="MobileNet" width=40% height=40%>
  <figcaption> MobileNet </figcaption>
</figure>
</p>


## Future work & Improvements
