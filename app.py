# import necessary libraries
import streamlit as st
import tensorflow as tf
import cv2
from PIL import Image, ImageOps
import numpy as np
import os
import pandas as pd
import numpy as np
import glob
import matplotlib.pyplot as plt
import seaborn as sns
from PIL import Image
import cv2
import pathlib
import random
from tqdm import tqdm
from random import sample
import itertools
import shutil

import math
import numpy as np
import tensorflow as tf
from tensorflow import keras
# import tensorflow_addons as tfa
import matplotlib.pyplot as plt
from tensorflow.keras import layers
# from fastai.vision import *
# from fastai.metrics import error_rate, accuracy

import tensorflow as tf
# import tensorflow_addons as tfa
# import torch
from tensorflow import keras

from tensorflow.keras.preprocessing.image import ImageDataGenerator
from keras.utils.np_utils import to_categorical
# from keras.optimizers import adam_v2
from tensorflow.keras.preprocessing import image
from keras.layers import Conv2D, BatchNormalization, MaxPool2D, Dropout, Flatten, Dense, SeparableConv2D, GlobalAveragePooling2D, Input, ZeroPadding2D
from tensorflow.keras.models import Model, Sequential, load_model
from tensorflow.keras.callbacks import ReduceLROnPlateau, ModelCheckpoint, EarlyStopping, LearningRateScheduler
from tensorflow.keras.optimizers import Adam, RMSprop, SGD
from tensorflow.keras import regularizers, layers, optimizers
from tensorflow.keras.applications import ResNet50V2, Xception, ResNet101V2, VGG16, EfficientNetV2S

from sklearn.model_selection import train_test_split, StratifiedShuffleSplit
# from imblearn.over_sampling import SMOTE
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, classification_report, recall_score


model = tf.keras.models.load_model('opacity_model.h5')
# import necessary libraries
import streamlit as st
import tensorflow as tf
import cv2
from PIL import Image, ImageOps
import numpy as np
st.set_option("deprecation.showfileUploaderEncoding",False)


# cache the app so that it runs faster
@st.cache(allow_output_mutation=True)


def loading_model():
  """
  Loading_model:
          Returns: the model weights that are stored in the path specified
    tf.keras.models.load_model('path to your model weights'(.h5 file))      

  """
  model = tf.keras.models.load_model('opacity_model.h5')
  return model

# Designing how the interface of the web page will look

with st.spinner("[INFO] Model loading ...."): #displays a temporary message as model is loading
  # call loading_model() to obtain the model
  model = loading_model()


# st.title('Omdena Uganda')
st.title('Opacity Classification')
st.write('  ')
st.subheader("This is an online platform for Opacity classification")


# allows user to upload images. it accepts only jpg and png types
file = st.file_uploader("Upload your image here ",
                        type=["jpg","png"])




def import_model_and_predict(image_data,model):

  """ Import_model_and_predict: uses the uploaded model to make prediction
      
      Inputs: 
        image_data: Input image that the model will make prediction
        model: saved model that will classify the input image
      Output:
          Return a prediction carried out on the input image

  """
  # image size which your model was trained on
  size = (150,150)
  # resize the input image to required size minimizing information loss
  image = ImageOps.fit(image_data,size,Image.ANTIALIAS)
  # converts PIL images to numpy arrays
  image = np.asarray(image)
  # convert color scheme from BGR to RGB
  img = cv2.cvtColor(image,cv2.COLOR_BGR2RGB)

  # introduce a 4th dimension to your image
  img_reshape = img[np.newaxis,...]
  # make prediction on the image
  prediction = model.predict(img_reshape)
  return prediction

if file is None:
  st.text("Please upload an image File")
else:
  image = Image.open(file)
  st.image(image,use_column_width=True)
  predictions = import_model_and_predict(image,model)
  score = tf.nn.softmax(predictions[0])
  # st.write(predictions)
  # st.write(score)
  class_names = ['normal', 'sick']
  string = "This image is: "+ class_names[np.argmax(predictions)]
  st.success(string)
 

  

  print("{} Disease detected with {:.2f} percent confidence".format(
      class_names[np.argmax(score)],100*np.max(score)
  ))
    

 
 
