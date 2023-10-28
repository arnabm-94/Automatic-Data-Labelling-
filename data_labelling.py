
#Python code to automate image labeling using a pre-trained VGG16 model from the Keras library.

###############################################################################

#In this section, we import the required libraries for working with TensorFlow and Keras, a popular deep learning library. 
#We also import the VGG16 model, which is a pre-trained model designed for image classification, as well as other necessary modules.

# Import necessary libraries
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.applications import VGG16
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.vgg16 import preprocess_input, decode_predictions
import numpy as np

#Load the pre-trained VGG16 model. 
#This model has already been trained on a large dataset for image classification tasks and is available through Keras. 
#By specifying weights='imagenet', we load the model with pre-trained weights for classifying images into 1,000 different categories.


# Load a pre-trained CNN model (VGG16 in this example)
model = VGG16(weights='imagenet')

#This section defines a function called label_image that takes the path to an image as input and returns the top predicted label for that image. 

#Step-by-step breakdown of what the function does:

#image.load_img(image_path, target_size=(224, 224)): This line loads the image from the specified path and resizes it to 224x224 pixels. VGG16 expects input images of this size.

#image.img_to_array(img): Converts the image to a NumPy array, which can be used as input to the model.

#np.expand_dims(img_array, axis=0): Adds an extra dimension to the array to match the input shape expected by the VGG16 model.

#preprocess_input(img_array): Preprocesses the input image array to match the preprocessing applied during the training of the VGG16 model.

#model.predict(img_array): Feeds the preprocessed image array to the VGG16 model, making predictions for the image.

#decode_predictions(predictions, top=1): Decodes the model's predictions to human-readable labels. The top=1 argument specifies that we want the top predicted label.

#The function returns the label of the top prediction as a human-readable class name.

# Function to predict and label an image
def label_image(image_path):
    img = image.load_img(image_path, target_size=(224, 224))
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array = preprocess_input(img_array)
    
    predictions = model.predict(img_array)
    decoded_predictions = decode_predictions(predictions, top=1)[0]
    
    # Return the top predicted label
    return decoded_predictions[0][1]

# Example usage
image_path = 'path_to_image.jpg'
predicted_label = label_image(image_path)
print(f'The predicted label for the image is: {predicted_label}')

#This part demonstrates how to use the label_image function. 
#You specify the path to an image, and the function returns the predicted label. 
#Finally, it prints the predicted label to the console.