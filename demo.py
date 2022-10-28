import streamlit as st
import keras
import numpy as np
import pandas as pd
import os
import io
import efficientnet.tfkeras as efc
import matplotlib.pyplot as plt
import matplotlib.pyplot as plt
import tensorflow.keras.backend as K

from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten
from keras.layers import Conv2D, MaxPooling2D
from keras.preprocessing import image
from sklearn.model_selection import train_test_split
from keras.layers import BatchNormalization
from tensorflow.keras.initializers import Constant
from tensorflow.keras.models import Model
from tensorflow.keras.layers import *
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from tensorflow.keras.optimizers import Adam, RMSprop
from tensorflow.keras import optimizers
from tensorflow.keras.models import load_model
from tensorflow.keras.callbacks import ReduceLROnPlateau
from tensorflow.keras.callbacks import LambdaCallback
from tensorflow.keras import preprocessing
from sklearn.metrics import classification_report, confusion_matrix
from PIL import Image
from sftp import SFTP

os.environ['CUDA_VISIBLE_DEVICES']='0'

sftp = SFTP(st.secrets["HOSTNAME"], st.secrets["USERNAME"], st.secrets["PASSWORD"])


def load_image():
    uploaded_file = st.file_uploader(label='Pick an image to test')
    if uploaded_file is not None:
        image_data = uploaded_file.getvalue()
        st.image(image_data)
        return Image.open(io.BytesIO(image_data))
    else:
        return None

    
def load_model():
    remoteFilePath = '/nas2/epark/safety_management_git/total_EfficientNetB4_revision.h5'
    localFilePath = 'total_EfficientNetB4_revision.h5'
    sftp.download(remoteFilePath, localFilePath)
    model = efc.EfficientNetB4(weights="imagenet", include_top=False, input_shape=(224, 224, 3))
    seed = 1200
    new_model = Sequential([
        model,
        keras.layers.GlobalAveragePooling2D(),
        keras.layers.Dense(512,
        kernel_initializer=keras.initializers.RandomUniform(seed=seed),
        bias_initializer=keras.initializers.Zeros(), name='dense', activation='tanh'),

         keras.layers.Dense(9,name='dense_top', activation='sigmoid')])

    new_model.load_weights(localFilePath)
    new_model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['acc'])

    return new_model


def predictLabelForGivenThreshold(threshold):
    y_predict=[]
    for sample in  Y_pred:
        y_predict.append([1 if i>=threshold else 0 for i in sample ] )
    return np.array(y_predict)


def predict(model, image):
    test_image = image.resize((224,224))
    test_image = preprocessing.image.img_to_array(test_image)
    test_image = test_image / 255.0
    test_image = np.expand_dims(test_image, axis=0)

    preds = model.predict(test_image)
    preds = preds.tolist()
    
    indices = []
    for pred in preds:
        temp = []
        for category in pred:
            if category>=0.6:
                temp.append(pred.index(category))
        if temp!=[]:
            indices.append(temp)
        else:
            temp.append(np.argmax(pred))
            indices.append(temp)

    st.write(indices)
 

def main():
    remoteFilePath2 = '/nas2/epark/safety_management_git/category_label.csv'
    localFilePath2 = 'category_label.csv'
    sftp.download(remoteFilePath2, localFilePath2)
    st.title('Safety management pretrained model demo')
    model = load_model()
    image = load_image()
    result = st.button('Run on image')
    if result:
        st.write('Predicting results...')
        predict(model, image)
    st.subheader("Class 정보")

    df = pd.read_csv(localFilePath2)
    st.write(df)


if __name__ == '__main__':
    main()