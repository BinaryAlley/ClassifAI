# region ================================================================== IMPORTS ====================================================================================
import cv2
import os.path
import requests
import validators
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from PIL import Image
from PIL import ImageFont
from PIL import ImageDraw 
from urllib.parse import urlparse
from requests.models import HTTPError
from tflite_model_maker import configs
from requests.exceptions import Timeout
from tflite_model_maker import image_classifier
from classification_model import ClassificationM
from tflite_model_maker import ImageClassifierDataLoader

from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.models import Sequential
# endregion

class DeepNeuralNetwork:
    """ Deep Neural Network for Flower Species Classification

    Implements a classifier using TensorFlow and Keras to identify flower species. It includes methods for training, testing, 
    predicting, and utility functions for image processing.

    Creation Date: 30th of December, 2020

    Attributes:
        model: TensorFlow/Keras model for classification
        testData: Dataset for testing the model
        trainData: Dataset for training the model
        validationData: Dataset for model validation
        isModelTrained: Boolean indicating if the model has been trained
        classifications: List of classification categories

    Methods:
        TrainKerasModel: Trains the model using Keras
        TrainDeepNeuralNetwork: Trains the model using TensorFlow Lite Model Maker
        TestTrainedDeepNeuralNetwork: Tests the trained model
        PredictFlowerClass: Predicts the flower class for a given image
        WriteResultsOnImage: Writes classification results on an image
        DownloadImage: Downloads an image from the web
        __GetLabelColor: Private helper function for UI purposes
    """
# region =============================================================== FIELD MEMBERS =================================================================================
    model = None 
    testData = None
    trainData = None
    validationData = None
    # make sure the tensorflow version is the right one
    print ('TensorFlow version ' + tf.__version__)
    # if the model file exists, no need to re-train it each time
    isModelTrained = os.path.exists(os.path.abspath(os.path.dirname(__file__)) + os.path.sep + 'export' + os.path.sep + 'model_quant.tflite')
    # declare tha classification categories for which the model was trained
    classifications = [
        ClassificationM('Daisy', 'Plantae', 'Tracheophytes', 'Angiosperms', 'Eudicots', 'Asterales', 'Asteraceae', 'Cichorioideae', 'Bellis', 'Bellis perennis'),
        ClassificationM('Dandelion', 'Plantae', 'Tracheophytes', 'Angiosperms', 'Eudicots', 'Asterales', 'Asteraceae', 'Cichorioideae', 'Cichorieae', 'Taraxacum'),
        ClassificationM('Rose', 'Plantae', 'Tracheophytes', 'Angiosperms', 'Eudicots', 'Rosales', 'Rosaceae', 'Rosoideae', 'Roseae', 'Rosa'),
        ClassificationM('Sunflower', 'Plantae', 'Tracheophytes', 'Angiosperms', 'Eudicots', 'Asterales', 'Asteraceae', 'Asteroideae', 'Heliantheae', 'Helianthus'),
        ClassificationM('Tulips', 'Plantae', 'Tracheophytes', 'Angiosperms', 'Monocots', 'Liliales', 'Liliaceae', 'Lilioideae', 'Lilieae', 'Tulipa')
    ]
# endregion
    
# region ==================================================================== CTOR =====================================================================================
    def __init__(self):
        """ Initializes the Deep Neural Network classifier

        Ensures TensorFlow version 2 is being used and configures TensorFlow to handle GPU memory growth;
        This setup is crucial for avoiding common issues in GPU-based TensorFlow operations

        Raises:
            AssertionError: If the TensorFlow version is not 2.x
        """
        assert tf.__version__.startswith('2')
        # solve the bug of tensorflow looking for cusolver64_10.dll when cusolver64_11.dll is the correct one installed (NVidia CUDA)
        for gpu in tf.config.experimental.list_physical_devices(device_type = 'GPU'):
            tf.config.experimental.set_memory_growth(gpu, True)
# endregion

# region ================================================================== METHODS ====================================================================================
    def TrainKerasModel(self):
        """
        Trains a Convolutional Neural Network (CNN) using TensorFlow's Keras API, for image classification
        """
        import pathlib
        # define the directory path for training data
        data_dir = pathlib.Path(os.path.abspath(os.path.dirname(__file__)) + os.path.sep + 'data' + os.path.sep + 'flower_photos')
        # set parameters for image processing, mandatory for ensuring consistent input tensor shape for the neural network
        
        # define hyperparameters for the model training
        # batch size impacts the stochastic gradient descent process by determining the number of samples to work through, before updating the internal model parameters
        batch_size = 32
        img_height = 180
        img_width = 180
        # load and preprocess training and validation datasets; the dataset is split into training (80%) and validation (20%) subsets; 'seed' ensures reproducibility of results
        train_ds = tf.keras.preprocessing.image_dataset_from_directory(data_dir, validation_split=0.2, subset="training", seed=123, image_size=(img_height, img_width), batch_size=batch_size)
        val_ds = tf.keras.preprocessing.image_dataset_from_directory(data_dir, validation_split=0.2, subset="validation", seed=123, image_size=(img_height, img_width), batch_size=batch_size)
        class_names = train_ds.class_names
        # AUTOTUNE dynamically adjusts the value for optimal throughput at runtime, beneficial for loading data efficiently
        AUTOTUNE = tf.data.experimental.AUTOTUNE
        # use data shuffling, it ensures that each data batch has a random sample of observations, reducing model overfitting risk
        # use caching and prefetching to reduce read latency while training the model
        train_ds = train_ds.cache().shuffle(1000).prefetch(buffer_size=AUTOTUNE)
        val_ds = val_ds.cache().prefetch(buffer_size=AUTOTUNE)
        # the number of neurons in the final Dense layer corresponds to the number of output classes (flower species)
        num_classes = 5
        # data augmentation through random rotation, flips and zooming introduces variability in the dataset, effectively increasing dataset size and diversity, 
        # simulating a large training dataset; this helps prevent overfitting and improves model generalization
        data_augmentation = keras.Sequential(
        [
            layers.experimental.preprocessing.RandomFlip("horizontal", input_shape=(img_height, img_width, 3)),
            layers.experimental.preprocessing.RandomRotation(0.1),
            layers.experimental.preprocessing.RandomZoom(0.1),
        ])
        # model architecture: a Convolutional Neural Network with Conv2D and MaxPooling2D layers:
        # Conv2D layers perform convolution operations, capturing spatial features in the image;
        # MaxPooling2D reduces the spatial dimensions (height and width) of the input volume for the next convolution layer;
        # Dropout layer randomly sets input units to 0, with a frequency of rate, at each step during training, which helps prevent overfitting    
        model = Sequential([
        data_augmentation,
        # rescaling layer normalizes the pixel values to the range [0, 1], which aids in stabilizing the training process
        layers.experimental.preprocessing.Rescaling(1./255),
        # Conv2D layer with 16 filters, each with a kernel size of 3x3.
        # Filters in convolutional layers are feature detectors. Here, 16 filters imply the layer will learn 16 distinct patterns.
        # The kernel size of 3x3 means each filter covers a 3x3 region of the input image to compute feature activation.
        # ReLU (Rectified Linear Unit) activation function is used.
        # ReLU is defined as f(x) = max(0, x). It introduces non-linearity, allowing the network to learn complex patterns.
        layers.Conv2D(16, 3, padding='same', activation='relu'),
        # MaxPooling2D reduces the spatial dimensions (height and width) of the input volume.
        # A 2x2 pooling layer reduces the height and width by a factor of 2.
        # This operation compresses the image and reduces computational load for subsequent layers
        layers.MaxPooling2D(),
        # increasing the number of filters in subsequent layers allows the network to capture more complex features;
        # a pattern in the initial layers might be simple edges, but in deeper layers, it could be textures or object parts
        layers.Conv2D(32, 3, padding='same', activation='relu'),
        layers.MaxPooling2D(),
        # 64 filters in this layer increase the depth of the network further, allowing it to learn even more complex features
        layers.Conv2D(64, 3, padding='same', activation='relu'),
        layers.MaxPooling2D(),
        # dropout layer randomly sets input units to 0 with a rate of 0.2 at each step during training time
        # This prevents overfitting by ensuring that the network does not rely too much on any single node.
        layers.Dropout(0.2),
        # flatten layer converts the 3D output from the previous layers into a 1D array; this transformation is necessary to feed the data into a Dense layer for classification
        layers.Flatten(),
        # Dense layer is a fully connected layer with 128 neurons.
        # It receives input from all neurons in the previous layer, processing it through a weighted sum, followed by a bias offset.
        # The ReLU activation function is applied so as to infuse non-linearity, such that the model is able to learn complex patterns
        layers.Dense(128, activation='relu'),
        # the final Dense layer outputs the predictions. It has a neuron for each class - the network will output a 5-dimensional vector, indicating the probability of each class
        layers.Dense(num_classes)
        ])
        # The model is compiled with the Adam optimizer, which is a stochastic gradient descent method that is based on adaptive estimation of first-order and second-order moments, 
        # and a loss function suitable for multi-class classification.
        # Mathematically, Adam combines the advantages of two other extensions of stochastic gradient descent: 
        # Adaptive Gradient Algorithm (AdaGrad) and Root Mean Square Propagation (RMSProp)
        # Adam optimizer formula: θ_{t+1} = θ_t - lr * m_t / (√v_t + ε), where:
        #   θ represents the parameters (weights)
        #   lr is the learning rate
        #   m_t is the first moment vector (mean)
        #   v_t is the second moment vector (uncentered variance)
        #   ε is a small scalar used for numerical stability.
        # Sparse Categorical Crossentropy is used as the loss function, as it's suitable for classifying multiple classes when each sample belongs to exactly one class
        model.compile(optimizer='adam',
              loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
              metrics=['accuracy'])
        # display the model's architecture
        model.summary()
        # number of complete passes through the training dataset
        epochs = 15
        # the model is trained for a fixed number of epochs, and learning is validated on a separate set
        history = model.fit(train_ds, validation_data=val_ds, epochs=epochs )   
        # plot training and validation metrics, to evaluate model performance;
        # training accuracy and loss demonstrate how well the model is learning, while validation accuracy and loss show how well the model generalizes to unseen data    
        acc = history.history['accuracy']
        val_acc = history.history['val_accuracy']
        loss = history.history['loss']
        val_loss = history.history['val_loss']

        epochs_range = range(epochs)

        plt.figure(figsize=(8, 8))
        plt.subplot(1, 2, 1)
        plt.plot(epochs_range, acc, label='Training Accuracy')
        plt.plot(epochs_range, val_acc, label='Validation Accuracy')
        plt.legend(loc='lower right')
        plt.title('Training and Validation Accuracy')

        plt.subplot(1, 2, 2)
        plt.plot(epochs_range, loss, label='Training Loss')
        plt.plot(epochs_range, val_loss, label='Validation Loss')
        plt.legend(loc='upper right')
        plt.title('Training and Validation Loss')
        plt.show()
        # save the trained model for future inference
        model.save('.' + os.path.sep + 'export' + os.path.sep + 'model')

    def TrainDeepNeuralNetwork(self):
        """ Trains a new Deep Neural Network classifier and exports the resulting model
        """
        # get the archive of images to be used as for the model training and save it on disk
        data_path = tf.keras.utils.get_file(os.path.abspath(os.path.dirname(__file__)) + os.path.sep + 'data' + os.path.sep + 'flower_photos',
            'https://storage.googleapis.com/download.tensorflow.org/example_images/flower_photos.tgz', untar = True)
        # load data from folder (jpg and png only!)
        data = ImageClassifierDataLoader.from_folder(data_path)
        # split data to training data (80%), validation data (10%, optional) and testing data (10%)
        self.trainData, rest_data = data.split(0.8)
        self.validationData, self.testData = rest_data.split(0.5)
        # create labels for categorical data
        label_names = data.index_to_label
        # load data and train a model of type EfficientNet-Lite0, with an input image shape of [224, 224]
        self.model = image_classifier.create(self.trainData, validation_data=self.validationData, epochs=10, batch_size=32)
        self.model.summary()
        # enforce full integer quantization for all ops including the input and output (not required, but recommended on mobile, significatively reduces model size)
        # config = configs.QuantizationConfig.create_full_integer_quantization(representative_data=self.testData, is_integer_only=True)
        # convert the existing model to TensorFlow Lite model format with metadata; use with_metadata = False to export labels to text file, instead of embedded in model
        # self.model.export(export_dir='.' + os.path.sep + 'export' + os.path.sep, tflite_filename='model_quant.tflite', label_filename='model_labels.txt', quantization_config=config, with_metadata=False)
        self.model.export(export_dir='.' + os.path.sep + 'export' + os.path.sep, tflite_filename='model_quant.tflite', with_metadata=True)
       
    def TestTrainedDeepNeuralNetwork(self):
        """ Test the trained Deep Neural Network on a batch of 50 images used in the training process
        """
        # make sure the mandatory data is populated
        if (self.model is None):
            raise TypeError("Model cannot be None! Please train the model first!")
        if (self.testData is None):
            raise TypeError("Test data cannot be None! Please train the model first!")
        # plot 50 test images and their predicted labels; if a prediction result is different from the provided label in "test" dataset, 
        # highlight it in red color.
        plt.figure(figsize = (8, 8)) # 8 x 8 inches figure
        # predict the classification of the top 50 items
        predicts = self.model.predict_top_k(self.testData, batch_size = 32)
        # cycle through the 50 test images
        for i, (image, label) in enumerate(self.testData.dataset.take(50)):
            # create a subplot of 10 rows x 5 columns and assign each iterated image to a cell
            ax = plt.subplot(10, 5, i + 1)
            # get the current values for the tick locations and labels of axis without modifying them
            plt.xticks([])
            plt.yticks([])
            # do not show grid lines
            plt.grid(False)
            # display data as an image on a 2D regular raster
            plt.imshow(image.numpy(), cmap=plt.cm.gray)
            # get the predicted classification label
            predict_label = predicts[i][0][0]
            # assign a color based on the prediction result (red if the prediction doesn't match the metadata, black otherwise)
            color = self.__GetLabelColor(predict_label, self.testData.index_to_label[label.numpy()])
            ax.xaxis.label.set_color(color)
            # add a label with the predicted classification
            plt.xlabel('Predicted: %s' % predict_label)
        # show the graph
        plt.show()

    def PredictFlowerClass(self, _path):
        """ Predicts the classification of the provided image using a trained Deep Neural Network model
        
        Parameters:
            _path (str): The path to the image for which to predict the classification

        Returns:
            - str: A string representing the recognized classification category, or None if not recognized
            - str: The accuracy of the prediction in percentage, as a formatted string
        
        """
        # check if the model exists
        if not os.path.exists(os.path.abspath(os.path.dirname(__file__)) + os.path.sep + 'export' + os.path.sep + 'model_quant.tflite'):
            raise FileNotFoundError("The Deep Neural Network model file does not exist!")
        # check if the file exists (locally or online)
        if not os.path.exists(_path) and not validators.url(_path):
            raise FileNotFoundError("The provided image must either be a local or an online file!")
        # check the provided file type
        if not _path.endswith('jpg') and not _path.endswith('jpeg') and not _path.endswith('png'):
            raise TypeError("Only JPEG and PNG images allowed!")
        # if an URL is supplied, download the image locally and used the saved file path
        if validators.url(_path):
            _path = self.DownloadImage(_path)
        # load the provided image image as a ndarray
        image_a = plt.imread(_path)
        # resize the image to a standard size of maximum 224 x 224 pixels
        image_a = cv2.resize(image_a, (224, 224))
        # convert the input data as a numpy array and normalize it to 0 - 1
        image_a = np.asarray(image_a) / 255
        # change the shape of the numpy array 
        image_a = np.reshape(image_a, (1, 224, 224, 3))
         # load the TFLite model
        interpreter = tf.lite.Interpreter(model_path = os.path.abspath(os.path.dirname(__file__)) + os.path.sep + 'export' + os.path.sep + 'model_quant.tflite')
        # allocate tensors to the model
        interpreter.allocate_tensors()
        # get the input and output tensors
        input_details = interpreter.get_input_details()
        output_details = interpreter.get_output_details()
        # test the model on the provided input data
        input_data = np.array(image_a, dtype = np.float32)
        # set the value of the input tensor
        interpreter.set_tensor(input_details[0]['index'], input_data)
        # get the input shape used in the model
        input_shape = input_details[0]['shape']
        # invoke the interpreter
        interpreter.invoke()
        # the function get_tensor() returns a copy of the tensor data; use tensor() in order to get a pointer to the tensor
        output_data = interpreter.get_tensor(output_details[0]['index'])
        # get the highest value in the output result
        prediction_highest_value = np.amax(output_data[0])
        # only consider valid recognition if the predicted value is higher than 80%
        if (prediction_highest_value > 0.80):
            # get the index of the numpy array element containing the highest prediction value
            prediction_label_index = np.where(output_data[0] == prediction_highest_value)[0][0]
            # write metadata on image
            self.WriteResultsOnImage(_path, self.classifications[prediction_label_index].name, str(round(prediction_highest_value * 100, 2)))
            # returned the predicted classification category and the accuracy of the prediction
            return self.classifications[prediction_label_index], str(round(prediction_highest_value * 100, 2)) + ' %'
        else:
            # image was not recognized in any of the classification categories
            return None

    def WriteResultsOnImage(self, _path, _class, _accuracy):
        """ Writes the classification category and prediction accuracy on an image
    
        Parameters:
            _path (str): The path to the image where the information will be written
            _class (str): The classification category to write on the image
            _accuracy (str): The prediction accuracy to write on the image        
        """
        # open the image to be modified
        my_image = Image.open(_path)
        # load a truetype font
        title_font = ImageFont.truetype('arial.ttf', 40)
        image_editable = ImageDraw.Draw(my_image)
        # draw a rectangle as the background of the text, avoid white text on possibly white images
        image_editable.rectangle([(0, 0), (600, 70)], fill = '#000000', outline = None)
        # draw the predicted classification and precission on the image
        image_editable.text((15, 15), _class + ", " + _accuracy + ' % accuracy', (237, 230, 211), font = title_font)
        # save the modified image
        my_image.save(_path)

    def DownloadImage(self, _path):
        """ Downloads an image from the web

        Parameters:
            _path (str): The URL of the image to download

        Returns:
            str: The local path of the downloaded image if successful, None if the download failed
        """
        try:
            # create a HTTP GET request
            response = requests.get(_path)
            # if HTTP response is not 200 (OK), raise exception
            response.raise_for_status()
            # save the file locallh
            file = open(os.path.abspath(os.path.dirname(__file__)) + os.path.sep + 'predicted' + os.path.sep + os.path.basename(urlparse(_path).path), "wb")
            file.write(response.content)
            file.close()
            # check that the file was created
            if (os.path.exists(os.path.abspath(os.path.dirname(__file__)) + os.path.sep + 'predicted' + os.path.sep + os.path.basename(urlparse(_path).path))):
                return os.path.abspath(os.path.dirname(__file__)) + os.path.sep + 'predicted' + os.path.sep + os.path.basename(urlparse(_path).path)
            else:
                # the saved file does not exist 
                return None
        # catch any exceptions that might have occured
        except ConnectionError as _conn_err:
            print("Connection error: {0}".format(_conn_err))
        except HTTPError as _http_err:
            print("HTTP error: {0}".format(_http_err))
        except Timeout as _timeout_ex:
            print("Timeout exception: {0}".format(_timeout_ex))
        except:
            print('Unknown error while trying to download the image!')
        return None

    def __GetLabelColor(self, _firstValue, _secondValue):
        """ Helper function that returns 'red'/'black' depending on whether its two input parameters match or not

        Parameters:
            _firstValue (str): The first value to compare
            _secondValue (str): The second value to compare

        Returns:
            str: 'red' if the input values do not match, 'black' if they match
        """
        if _firstValue == _secondValue:
            return 'black'
        else:
            return 'red'
# endregion