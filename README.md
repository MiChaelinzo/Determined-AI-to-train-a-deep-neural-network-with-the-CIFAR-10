# Determined-AI-to-train-a-deep-neural-network-on-the-CIFAR-10

This code uses the CIFAR-10 dataset, a common benchmark dataset for image classification. The model architecture is a convolutional neural network that consists of two convolutional layers, max-pooling layers, and two fully connected layers. The data loader function preprocesses the data and creates TensorFlow Datasets for training and validation. The training function uses the Determined AI library to distribute the training across multiple GPUs using the TensorFlow backend. The hyperparameters, such as batch size, learning rate, and number of epochs, are defined in the experiment configuration, and the searcher selects the best set of hyperparameters based on the validation accuracy. The resources section specifies the number of slots per trial, which determines the number of GPUs used for each trial. Finally, the experiment is created using the Determined object, which manages the training process and outputs the results.



