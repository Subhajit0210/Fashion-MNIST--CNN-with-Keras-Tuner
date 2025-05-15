# Fashion-MNIST--CNN-with-Keras-Tuner

## Table of Contents
- [Project Overview](#project-overview)
- [Dependencies](#dependencies)
- [Data Collection](#data-collection)
- [Data Preprocessing](#data-preprocessing)
- [Data Collection](#data-collection)
- [Creating and Saving Augmented Samples](#creating-and-saving-augmented-samples)
- [Reference and Inspiration](#reference-and-inspiration)
- [Usage](#usage)
- [Contributing](#contributing)

## Project Overview
The Fashion MNIST dataset is a collection of 70,000 grayscale images of 10 different clothing items. This project implements a CNN model to classify these images. Keras Tuner's `RandomSearch` algorithm is employed to find the best architecture and hyperparameters for CNN.

## Dependencies
The following libraries are required to run this project:
- NumPy
- TensorFlow
- Keras
- Keras Tuner

## Data Collection
The Fashion MNIST dataset is automatically downloaded and loaded directly from the `keras.datasets` module. It consists of:
- **Training Set:** 60,000 images and their corresponding labels.
- **Test Set:** 10,000 images and their corresponding labels.

Each image is a 28x28 pixel grayscale image representing one of 10 clothing categories (e.g., T-shirt/top, trousers, pullover, dress, coat, sandal, shirt, sneaker, bag, ankle boot).

## Data Preprocessing
The raw image data from the Fashion MNIST dataset is preprocessed to make it suitable for input into the CNN model. The preprocessing steps include:
-  **Normalization:** The pixel values of the grayscale images, which are originally in the range of 0 to 255, are scaled down to a range of 0 to 1. This is achieved by dividing each pixel value by 255.0. Normalization helps improve model performance and training speed by ensuring that all features are on a similar scale.
-  **Reshaping:** The images are reshaped to have a 4-dimensional shape `(number_of_images, height, width, channels)`. For the Fashion MNIST dataset, which consists of grayscale images, the shape becomes `(number_of_images, 28, 28, 1)`. This reshaping is crucial because CNNs typically expect input data in this format to apply convolutional filters and extract spatial features effectively. The added dimension for 'channels' (1 for grayscale) is necessary for compatibility with the convolutional layers.


## Building the CNN Model Structure

The CNN model is built using the Keras Sequential API, which allows for creating a linear stack of layers [1]. The `build_model` function defines the model structure and incorporates hyperparameters that will be tuned by Keras Tuner.

The model consists of the following layers:

1.  **Convolutional Layers (`Conv2D`):**
    - These layers apply convolutional filters to the input images to extract features such as edges and textures.
    - The number of filters and the kernel size are defined as hyperparameters to be tuned by Keras Tuner using `hp.Int` and `hp.Choice` respectively.
    - The ReLU activation function is used in these layers to introduce non-linearity.
    - The first convolutional layer specifies the `input_shape` of the data `(28, 28, 1)`.

2.  **Flatten Layer (`Flatten`):**
    - This layer transforms the multi-dimensional output of the convolutional layers into a one-dimensional vector. This is necessary to connect the convolutional part of the network to the dense layers.

3.  **Dense Layers (`Dense`):**
    - These are fully connected layers where each neuron in the layer is connected to every neuron in the previous layer.
    - The number of units in the first dense layer is defined as a hyperparameter to be tuned using `hp.Int`.
    - The ReLU activation function is used in the first dense layer.
    - The final dense layer has 10 units (corresponding to the 10 clothing classes) and uses the softmax activation function. The softmax function outputs a probability distribution over the classes, indicating the likelihood of the input image belonging to each class.

## Compiling the Model

After defining the layers, the model is compiled using `model.compile()`. This step configures the model for training by specifying:

-   **Optimizer:** The optimization algorithm used to update the model's weights during training. The Adam optimizer is used, and its learning rate is also defined as a hyperparameter to be tuned using `hp.Choice`.
-   **Loss Function:** The function that measures the difference between the model's predictions and the true labels. `sparse_categorical_crossentropy` is used, which is suitable for multi-class classification with integer labels.
-   **Metrics:** The metrics used to evaluate the model's performance during training and testing. `accuracy` is used to measure the percentage of correctly classified images.

The `build_model` function returns the compiled model, ready for hyperparameter tuning and training.













































Fashion MNIST Dataset - https://www.kaggle.com/datasets/zalando-research/fashionmnist

Keras Tuner Blog - https://keras.io/keras_tuner/
