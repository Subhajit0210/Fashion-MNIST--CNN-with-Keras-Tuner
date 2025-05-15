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
1.  **Normalization:** The pixel values of the grayscale images, which are originally in the range of 0 to 255, are scaled down to a range of 0 to 1. This is achieved by dividing each pixel value by 255.0. Normalization helps improve model performance and training speed by ensuring that all features are on a similar scale.
2.  **Reshaping:** The images are reshaped to have a 4-dimensional shape `(number_of_images, height, width, channels)`. For the Fashion MNIST dataset, which consists of grayscale images, the shape becomes `(number_of_images, 28, 28, 1)`. This reshaping is crucial because CNNs typically expect input data in this format to apply convolutional filters and extract spatial features effectively. The added dimension for 'channels' (1 for grayscale) is necessary for compatibility with the convolutional layers.


Fashion MNIST Dataset - https://www.kaggle.com/datasets/zalando-research/fashionmnist

Keras Tuner Blog - https://keras.io/keras_tuner/
