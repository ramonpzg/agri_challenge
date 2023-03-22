# ML Project Discussion

This document outlines the possible routes I would follow in order to train a model that detects different kinds of spores floating around several farms across Australia and elswhere. The TL;DR section below answers the task in one paragraph, as requested, and the subsequent sections go into greater detail on the approaches I would take for this challenge.

## Table of Contents

1. [Task](##1.-Task)
2. [TL;DR](##2.-TL;DR)
3. [Project Scoping](##3.-Project-Scoping)
4. [Data Engineering](##4.-Data-Engineering)
5. [ML Model Development](##5.-ML-Model-Development)
   1. [Approach 1](###1.-Approach-1)
   2. [Approach 2](##2.-Approach-2)
   3. [Approach 3](###3.-Approach-3)
6. [Deployment](##6.-Deployment)
7. [Monitoring and Continual Learning](##7.-Monitoring-and-Continual-Learning)
8. [Business Analysis](##8.-Business-Analysis)

## 1. Task

<img title="" src="images/raw_input.jpg" alt="AJKFEHASIEFHAUISDHFwrfgasfdgsdf" data-align="center" width="461">

> In this question, we would like you to provide a **discussion paragraph** detailing how you would go about setting up a machine learning experiment to identify a particular spore type. Assume that a labelled dataset has been provided for you, including microscope images and labels of bounding boxes. Please outline the core tasks that would need to be completed in order to develop a machine learning model for this dataset.

<img title="" src="images/example_predictions.png" alt="AJKFEHASIEFHAUISDHFwrfgasfdgsdf" width="462" data-align="center">

## 2. TL;DR

Assuming that the data gets uploaded into a data lake for which I have access rights, the next step would be to develop a pipeline locally before scalling to the cloud. This pipeline would be built using MetaFlow and it would consist of a data loader and a combination of models with an experiment tracking tool (e.g. wandb) attached to it. The best model(s) would be saved into our data lake (with a git tag attached to it, or to whichever registry we are using for our models), and it will be served as a serverless API using AWS via bentoml. The series of models would be trained using PyTorch or FastAI and they would involve a (1) semantic or instance segmentation model to identify images with spores of a given deceace. This would allow us to use the output of such a model to focus on the areas of the image that contain deceases rather than large areas with nothing on them or with regular spores. In addition, it will help us balance the limited input of images with a decease as we exclude irrelevant input. Nexr (2) we would crop high focus areas and combine them into an image that would be used to build a (3) classification model that predicts the class of the spores in the image. To pain a better picture regarding this step, imagine taking the all the squares from y=(0, 2048) and x=(0, 1024) above and combining it with different squares. The last step would be to evaluate the results against the desired metric for the project, and, if happy, deploying our solution to start making predictions.

Here's a sketch of the process highlighted above.

```mermaid
graph LR
    A[(Labelled Data)] --> B(Train)
    A[(Labelled Data)] --> F(Validation)
    A[(Labelled Data)] --> G(Test)
    B --> C{Segmentation Model}
    F --> C{Segmentation Model}
    C --> D[Grouping]
    D --> E[Classification Model]
    E --> I[Experimentation]
    I --> E[Classification Model]
    G --> H[Final Model]
    E --> H[Final Model]
    H --> J(Business Analysis)
```

Here's a skeleton of the flow that would be run using MetaFlow.

```python
from metaflow import FlowSpec, Parameter, step, batch, conda, S3

class SporesClassifier(FlowSpec):

    S3_URI = Parameter(...)
    DATA_ROOT = Parameter(..., help='The local dir')
    IMAGES = Parameter(...)
    ANNOTATIONS = Parameter(...)
    PATH_TO_CONFIG = Parameter(...)
    ...

    @step
    def start(self):
        # Configure the (remote) experiment tracking location.
        import ...
        self.next(self.data_loader_from_s3)
    
    @step
    def data_loader_from_s3():
        """Get and Prep the Data"""
        self.dls = SegmentationDataLoaders.from_(...)
        self.next(self.train)

    
    @batch(gpu=1, memory=32000, image='gpu-latest', shared_memory=8000)
    @conda({"fastai": "2.7.11"})
    @step
    def train_unet(self):
        from fastai.vision.all import *
        self.train_args = dict("...": ...)
        self.learn = unet_learner(self.dls, **self.train_args)
        self.learn.fine_tune(8)
        self.next(self.grouping)

    @batch(cpu=6, memory=16000)
    @step
    def grouping(self):
        ...
        self.next(self.end)
    
    @batch(gpu=1, memory=32000, image='gpu-latest')
    @conda({"fastai": "2.7.11"})
    @step
    def train_cnn(self):
        from fastai.vision.all import *
        self.next(self.evaluate)

    @batch(cpu=4, memory=8000)
    @step
    def evaluate(self):
        ...
        bentoml.fastai.save_model(...)
        self.next(self.end)

    @step
    def end(self):
        """Analysis"""
        pass

if __name__ == '__main__':
    SporesClassifier()
```



## 3. Project Scoping

It is important to lay out the foundation for our project before we begin. This foundation will include overall goal of the project, metrics to optimize for, potential challenges, and more. Let's highlight these for our project.

- Goal: To reduce the amount of human labor required to label microscope-taken images with dreadful spores in them.

- Main Metric: Recall and, potentially, Specificity since we want to maximize the amount of true predictions in the dataset and minimize cost farmer's incurr when adding unnecessary pesticide.

- Potential challenges: highly inbalanced dataset.

## 4. Data Engineering

In this step, we would evaluate the shape and form of the images ingested by our models and create pipelines to have the images ready-to-go for training at any given moment. In addition, these pipelines would, ideally, be orchestrated in different ways, for example, to run every evening at midnight on VMs with different hardware requirements. For the orchestration layer we would use either an open source tool like MetaFlow, or a cloud-native one like Amazon Glue



Why different hardware requirements? If different pieces of hardware are down for maintenance for a day or two, that means that our pipeline will process less images and, therefore, would need less resources.



Note: Orchestrators and schedulers differ in one important way, resources. While orchestrators can shchedule, queue, and manage the resources a series of jobs would need, a scheduler cannot.

## 5. ML Model Development

Our model development would consist of a few parts:

1. Data Loaders: Assuming we will use PyTorch or FastAI, our data loader would be the piece of our model development pipeline that we connect to the output of our ETL or ELT pipeline from the previous method. Ideally, in this step we will manage train/valid/test split and other transformations necessary. For example, for we would take our spores annotated images for our task and split it 60-20-20.

2. Model or Series of Models: While we would most-likely try one model at a time for inference, it is normal to test approaches that chain multiple models together, as discussed in the TL;DR section. 

3. Experimentation Approach: In this stage, we want to try a combination of hyperparameters (e.g. learning rate) using a tool like Optuna, with an experiment tracking tool like Weights & Biases or MLFlow. That way we won't forget which combinations we ran and which was the best.

4. Model Store: Our models should be kept in an organized way so that we can access them easily in case we need to role back to a previous version or select a newer and better one. Weights & Biases provides a way to save artifacts in it, and another (somewhat preferred) way of saving a model would be via bentoml.

### 5.1 Approach 1

Pure classification, identifying the classes available

### 5.2 Approach 2

segmentation > grouping > classification > analysis

### 5.3 Approach 3

Augmentation

## 6. Deployment

AWS

BentoML (requires that docker terraform)

## 7. Monitoring and Continual Learning

Alibi detect

## 8. Business Analysis

Analysis of the classification output

## 9. Conclusion
