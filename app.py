import os
import keras
from keras.models import load_model
import streamlit as st
import tensorflow as tf
import numpy as np


st.header("Image Recognization CNN Model")
flower_names = ['daisy','dandelion','rose','sunflower','tulip']

model = load_model("C:\AI_Projects\Image_Recog_Model\Image_Recog_DP_model\Image_Recognisation_Model.h5")

#creating function
def classify_images(image_path):
    input_image = tf.keras.utils.load_img(image_path,target_size=(180,180))
    input_image_array = tf.keras.utils.img_to_array(input_image)
    input_image_exp_dim = tf.expand_dims(input_image_array, 0)

    prediction = model.predict(input_image_exp_dim) 
    
    result = tf.nn.softmax(prediction[0])
    outcome  = 'The Image belongs to  ' + flower_names[np.argmax(result )] + ' with a score of ' + str(np.max(result)*100)
    return outcome
    
upload_file = st.file_uploader("upload an image")
if upload_file is not None:
    with open(os.path.join('upload',upload_file.name), 'wb') as f:
        f.write(upload_file.getbuffer())
        
    st.image(upload_file, width=200)
    
st.markdown(classify_images(upload_file))