# Determined-AI-to-train-a-deep-neural-network-on-the-CIFAR-10

#### The project objective

This code uses the CIFAR-10 dataset, a common benchmark dataset for image classification. The model architecture is a convolutional neural network that consists of two convolutional layers, max-pooling layers, and two fully connected layers. The data loader function preprocesses the data and creates TensorFlow Datasets for training and validation. The training function uses the Determined AI library to distribute the training across multiple GPUs using the TensorFlow backend. The hyperparameters, such as batch size, learning rate, and number of epochs, are defined in the experiment configuration, and the searcher selects the best set of hyperparameters based on the validation accuracy. The resources section specifies the number of slots per trial, which determines the number of GPUs used for each trial. Finally, the experiment is created using the Determined object, which manages the training process and outputs the results.

#### A data sample from your dataset with an explanation

Json sample represents a hypothetical record in a medical dataset, with information about a patient who has been diagnosed with diabetes and is receiving insulin treatment. The record includes demographic information such as the patient's age and gender, as well as clinical information such as blood sugar levels, heart rate, and blood pressure measurements. The patient's admission and discharge dates are also included. This type of data can be used for a variety of medical research and analysis purposes, such as identifying trends in patient outcomes, predicting complications, or evaluating the effectiveness of different treatments. It is important to note that this is a hypothetical example and not based on any real medical data.

#### A description of your model architecture

In the context of distributed training with the Determined AI platform, the specific architecture used for the CIFAR-10 image classification task may have been optimized for performance and scalability across multiple nodes. The platform may have also used techniques such as data parallelism or model parallelism to distribute the computation across multiple GPUs or nodes.

- The first layer in a CNN is typically a convolutional layer, which applies a set of filters to the input image to extract feature maps. The output of the convolutional layer is then passed through a nonlinear activation function such as ReLU, which introduces nonlinearity into the model and helps to create more complex representations of the input data.

- The output of the activation function is then typically passed through a pooling layer, which reduces the spatial dimensions of the feature maps while retaining their essential features. This process is repeated multiple times, with the number of filters and the size of the filters gradually increasing to capture more complex features of the input image.

- Finally, the output of the last convolutional layer is flattened into a vector and passed through one or more fully connected layers, which use the extracted features to make a prediction about the class of the input image.

#### Instructions for how to run your training job

Install the Determined AI platform: Follow the instructions provided in the platform's documentation to install the Determined AI platform on your system.

1.) Prepare the CIFAR-10 dataset: Download the CIFAR-10 dataset and prepare it for training. This may involve preprocessing the images, splitting the dataset into training and validation sets, and converting the data into a format that can be read by the deep learning framework you will be using.

2.) Define your model: Using your preferred deep learning framework (such as TensorFlow or PyTorch), define your neural network architecture for the CIFAR-10 image classification task. This may involve specifying the number and size of convolutional layers, pooling layers, and fully connected layers, as well as the activation functions and loss function.

3.) Define your training job: Using the Determined AI platform, define your training job by specifying the number of nodes, the number of GPUs per node, and other training parameters such as the learning rate and batch size. You can also specify any hyperparameters you want to optimize during training, such as the number of layers or the size of the convolutional filters.

4.) Launch your training job: Once you have defined your training job, launch it using the Determined AI platform. The platform will distribute the training across the specified nodes and GPUs, allowing you to train your model at scale.

To run this job, save the above YAML file to a location on your machine and run the following command in the terminal:

lua
` det experiment create /path/to/yaml/file.yaml /path/to/project/directory` 

This command will launch a new training job on your machine or cluster using the specified YAML configuration file. You can monitor the progress of your job in real-time using the Determined AI web UI or CLI tool, and view the training metrics and evaluation results once the job is complete.

- This model consists of a series of convolutional layers followed by fully connected layers. The input to the network is a 32x32x3 image, and the output is a vector of length 10, representing the predicted probabilities for each of the 10 classes in the CIFAR-10 dataset.

You can use this model in your own code by importing the CIFAR10Model class and instantiating an instance of the class, as shown below:

python
```
from cifar10_model import CIFAR10Model

model = CIFAR10Model()
```
Then you can use this model instance to train and test on the CIFAR-10 dataset.




#### To run this application, you will need to have the following software and libraries installed:

Python 3.6 or later
Determined AI CLI (version 0.16.1 or later)
TensorFlow (version 2.0 or later)
Once you have installed the required software and libraries, follow these steps:

1.) Clone the GitHub repository to your local machine:
`git clone https://github.com/MiChaelinzo/Determined-AI-to-train-a-deep-neural-network-on-the-CIFAR-10.git`

2.) Navigate to the repository directory:

`cd Determined-AI-to-train-a-deep-neural-network-on-the-CIFAR-10 `

3.) Create a new Determined experiment using the CLI:

` det experiment create . --config=config.yaml` 

4.) Monitor the experiment using the CLI:

`det experiment describe <experiment_id> `

Replace <experiment_id> with the ID of the experiment created in step 3.

5.) Once the experiment is finished, you can view the results using the CLI:

`det experiment describe <experiment_id> --json | jq '.best_validation_metrics'`

Replace <experiment_id> with the ID of the experiment created in step 3.

- This will show the best validation accuracy achieved during the experiment.

#### To run the application, you can follow the steps below:

1.) Clone the repository:

` git clone https://github.com/MiChaelinzo/Determined-AI-to-train-a-deep-neural-network-on-the-CIFAR-10.git` 

2.) Install the required dependencies. The easiest way to do this is by creating a virtual environment and installing the dependencies using pip. You can create a virtual environment using the following command:

` python3 -m venv env `

Activate the virtual environment:

`source env/bin/activate `

Install the dependencies:

`pip install -r requirements.txt` 

Download the CIFAR-10 dataset:

` python download_dataset.py` 

Run the experiment:

`determined run experiment_config.yaml .` 

This will start the experiment using the configuration specified in experiment_config.yaml.

5.) Once the experiment is finished, you can view the results using the following command:

` determined tensorboard <experiment-id> `
Replace <experiment-id> with the ID of the experiment, which can be found in the output of the determined run command.

- Note: You will need to have a Determined cluster running to run this application. If you don't have a cluster, you can sign up for a free trial at https://determined.ai/. Once you have a cluster set up, you will need to update the determined.yaml file to point to the address of your cluster.













