# Fashion-MNIST--CNN-with-Keras-Tuner

## Table of Contents
- [Project Overview](#project-overview)
- [Dependencies](#dependencies)
- [Preparing the Save Location](#preparing-the-save-location)
- [Image Augmentation Configuration](#image-augmentation-configuration)
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



Fashion MNIST Dataset - https://www.kaggle.com/datasets/zalando-research/fashionmnist

Keras Tuner Blog - https://keras.io/keras_tuner/
