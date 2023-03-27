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

#### To run the cifar10_determined.py script from the GitHub repository MiChaelinzo/Determined-AI-to-train-a-deep-neural-network-with-the-CIFAR-10, you can follow these steps:

1.) Clone or download the repository to your local machine. 

2.) Ensure that you have the required dependencies installed. The script requires the following Python packages:

- determined (>=0.15.2)
- numpy
- torch (>=1.5.0)
- torchvision
- You can install them using pip: ` pip3 install -r requirements.txt` 

3.) Open a terminal or command prompt and navigate to the root directory of the downloaded repository.

Run the script using the following command:

` python3 cifar10_cnn_keras_tensor.py ` or ` python3 my_cnn_model.py ` depending on your needs

This command starts a Determined experiment and runs the cifar10_determined.py script using the current directory as the experiment directory.

Wait for the experiment to complete. The script will train a deep neural network on the CIFAR-10 dataset and print the training and validation accuracy for each epoch. Once the experiment is complete, the final validation accuracy will be displayed. You can also find the experiment logs, checkpoints, and other output in the experiment directory.


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

- Test data and the test accuracy/loss with cifar10_cnn_keras_tensor.py

<img width="562" alt="Screenshot 2023-03-14 153300" src="https://user-images.githubusercontent.com/68110223/227875797-f162989f-5890-4a77-8d26-03adb14a13d2.png">

- 100 epoch results from my_cnn_model.py 

<img width="773" alt="Screenshot 2023-03-26 125443" src="https://user-images.githubusercontent.com/68110223/227874451-3f5c86cf-72a1-4c11-9a54-5f0c2f30f6db.png">

- Testing out cifar10_model.h5 with tensorflow models (1)

<img width="691" alt="Screenshot 2023-03-25 203054" src="https://user-images.githubusercontent.com/68110223/227875186-17e53033-87d7-42fa-b6ef-236bd55c9c5e.png">

- Testing out cifar10_model.h5 with tensorflow models (2)

<img width="683" alt="Screenshot 2023-03-25 203114" src="https://user-images.githubusercontent.com/68110223/227875264-8de5e1c4-655a-4675-a738-b4d8f9422c3b.png">

- Result of Test Accuracy with cifar10_cnn.py (1)

<img width="676" alt="Screenshot 2023-03-13 132551" src="https://user-images.githubusercontent.com/68110223/224991555-f3cef14c-f59b-418b-b045-232bab2376dd.png">

- Result of Test Accuracy with cifar10_cnn.py (2)

<img width="653" alt="Screenshot 2023-03-14 151344" src="https://user-images.githubusercontent.com/68110223/227874142-66c22c58-d297-4749-a9a2-4d9a22103436.png">

 - Downloading Cifar 10 datasets
  
<img width="515" alt="Screenshot 2023-03-13 133257" src="https://user-images.githubusercontent.com/68110223/224991627-f2b40d90-6784-40d5-b62a-b69f0c91119f.png">

- Result of training cifar10_training.py (1)
  
<img width="907" alt="Screenshot 2023-03-13 142231" src="https://user-images.githubusercontent.com/68110223/224991973-b0ff9b5d-3577-47de-8062-468eb39ac7e3.png">

- Result of training cifar10_training.py (2)

  
<img width="928" alt="Screenshot 2023-03-13 142517" src="https://user-images.githubusercontent.com/68110223/224992095-a53230ee-33fc-4a02-a708-2b8d5583cd7f.png">

- Result of training and finished training cifar10_training.py (3)

  
<img width="670" alt="Screenshot 2023-03-13 142304" src="https://user-images.githubusercontent.com/68110223/224992151-550191cc-746e-4117-8355-fa9c067c19a1.png">

 - Result of accuracy and loss with cifar10_cnn_keras.py 
 ```
Epoch 1/10
2023-03-14 12:05:13.595124: I tensorflow/stream_executor/cuda/cuda_dnn.cc:368] Loaded cuDNN version 8200
1563/1563 [==============================] - 8s 4ms/step - loss: 1.5339 - accuracy: 0.4412 - val_loss: 1.2666 - val_accuracy: 0.5498
Epoch 2/10
1563/1563 [==============================] - 5s 3ms/step - loss: 1.1814 - accuracy: 0.5818 - val_loss: 1.1099 - val_accuracy: 0.6002
Epoch 3/10
1563/1563 [==============================] - 5s 3ms/step - loss: 1.0317 - accuracy: 0.6362 - val_loss: 1.0588 - val_accuracy: 0.6323
Epoch 4/10
1563/1563 [==============================] - 5s 3ms/step - loss: 0.9303 - accuracy: 0.6751 - val_loss: 0.9707 - val_accuracy: 0.6597
Epoch 5/10
1563/1563 [==============================] - 5s 3ms/step - loss: 0.8563 - accuracy: 0.6997 - val_loss: 0.9099 - val_accuracy: 0.6797
Epoch 6/10
1563/1563 [==============================] - 5s 3ms/step - loss: 0.7942 - accuracy: 0.7202 - val_loss: 0.8971 - val_accuracy: 0.6881
Epoch 7/10
1563/1563 [==============================] - 5s 3ms/step - loss: 0.7456 - accuracy: 0.7397 - val_loss: 0.8549 - val_accuracy: 0.7087
Epoch 8/10
1563/1563 [==============================] - 5s 3ms/step - loss: 0.6991 - accuracy: 0.7547 - val_loss: 0.8789 - val_accuracy: 0.7035
Epoch 9/10
1563/1563 [==============================] - 5s 3ms/step - loss: 0.6594 - accuracy: 0.7664 - val_loss: 0.8856 - val_accuracy: 0.7050
Epoch 10/10
1563/1563 [==============================] - 5s 3ms/step - loss: 0.6204 - accuracy: 0.7830 - val_loss: 0.8984 - val_accuracy: 0.6952
313/313 - 1s - loss: 0.8984 - accuracy: 0.6952 - 563ms/epoch - 2ms/step<img width="581" alt="Screenshot 2023-03-14 154703" src="https://user-images.githubusercontent.com/68110223/225050943-63cbc7a0-bed6-4ebc-aa94-becaa9ca4d08.png">

Test accuracy: 0.6952000260353088
```
<img width="581" alt="Screenshot 2023-03-14 154703" src="https://user-images.githubusercontent.com/68110223/225051119-497fab54-f456-4894-9616-46dffb52579d.png">
