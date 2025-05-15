# Fashion-MNIST--CNN-with-Keras-Tuner

## Table of Contents
- [Project Overview](#project-overview)
- [Dependencies](#dependencies)
- [Data Collection](#data-collection)
- [Data Preprocessing](#data-preprocessing)
- [CNN Model Structure](#cnn-model-structure)
- [Compiling the Model](#compiling-the-model)
- [Results and Insights](#results-and-insights)
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

Link to the dataset -  [Fashion MNIST Dataset](https://www.kaggle.com/datasets/zalando-research/fashionmnist)

## Data Preprocessing
The raw image data from the Fashion MNIST dataset is preprocessed to make it suitable for input into the CNN model. The preprocessing steps include:
-  **Normalization:** The pixel values of the grayscale images, which are originally in the range of 0 to 255, are scaled down to a range of 0 to 1. This is achieved by dividing each pixel value by 255.0. Normalization helps improve model performance and training speed by ensuring that all features are on a similar scale.
-  **Reshaping:** The images are reshaped to have a 4-dimensional shape `(number_of_images, height, width, channels)`. For the Fashion MNIST dataset, which consists of grayscale images, the shape becomes `(number_of_images, 28, 28, 1)`. This reshaping is crucial because CNNs typically expect input data in this format to apply convolutional filters and extract spatial features effectively. The added dimension for 'channels' (1 for grayscale) is necessary for compatibility with the convolutional layers.


## CNN Model Structure
The CNN model is built using the Keras Sequential API and defined within the `build_model` function. Hyperparameters for the model layers and optimizer are exposed for tuning by Keras Tuner.

The model includes:
-   **Convolutional Layers (`Conv2D`):** Extract features using tunable filters and kernel sizes. ReLU activation is applied.
-   **Flatten Layer (`Flatten`):** Converts the multi-dimensional convolutional output to a 1D vector.
-   **Dense Layers (`Dense`):** Fully connected layers for classification. The number of units in the first dense layer is tunable. The final dense layer has 10 units with a Softmax activation for class probability prediction.

## Compiling the Model
The model is configured for training using `model.compile()`. This involves:
-   **Optimizer:** Specifies the Adam optimizer with a tunable learning rate.
-   **Loss Function:** Uses `sparse_categorical_crossentropy` for multi-class classification.
-   **Metrics:** Monitors `accuracy` during training.

The `build_model` function returns the compiled model.

## Results and Insights

After running the Hyperparameter search using Keras Tuner, the results are stored in the specified output directory. You can analyze these results to understand:
-   **Best Hyperparameter Combination:** Keras Tuner identifies the combination of hyperparameters that yielded the best performance based on the specified objective (validation accuracy in this case).
-   **Performance of Different Configurations:** Review the performance metrics for each hyperparameter combination explored during the search. This provides insights into how different hyperparameter choices impact the model's performance.
-   **Model Training Progress:** The training process for the best model shows the convergence of loss and accuracy over epochs. Observing the training and validation curves can help identify potential overfitting or underfitting issues.

The code retrieves the best model found by the tuner and trains it for additional epochs. The summary of the best model architecture is also printed.

## Reference and Inspiration
This repository is based on the concepts presented in the Keras Tuner blog post.
Link to the blog - [Keras Tuner Blog](https://keras.io/keras_tuner/)
