## CIFAR-10 Image Classification using CNN

This project implements a Convolutional Neural Network (CNN) for image classification using the CIFAR-10 dataset with TensorFlow and Keras.

### Project Overview

The CIFAR-10 dataset consists of 60,000 32x32 color images belonging to 10 classes (airplanes, cars, birds, etc.). This project trains a CNN model to classify these images into their respective categories.

### Dependencies

* Python 3.8.0
* TensorFlow 
* Keras

### Running the Project

1. **Install dependencies:** Ensure you have Python and the required libraries installed. You can install them using pip:

   ```bash
   pip install tensorflow keras
   ```

2. **Run the cell:** Execute the jupyter cells containing your model definition, training, and evaluation code.

### Project Structure

* **Naan Mudhalvan AI project.ipynb:** This file contains the core implementation of the CNN model, including data loading, model definition, training, and evaluation.

### Data

The CIFAR-10 dataset is automatically downloaded by Keras during training.

### Model

The project utilizes a CNN architecture with convolutional layers, pooling layers, and fully connected layers for classification. You can modify the model architecture in `Naan Mudhalvan AI project.ipynb` to experiment with different configurations.

### Training

The model is trained on the CIFAR-10 training set. You can adjust hyperparameters like learning rate, epochs, and batch size in the file.

### Evaluation

The model's performance is evaluated on the CIFAR-10 test set. The script typically reports metrics like accuracy and loss.
