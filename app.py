import streamlit as st
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import numpy as np


MODEL_PATH = 'cnn_classifier.h5'
model = load_model(MODEL_PATH)

class_names = ['aarav','anu','malu','rohith','vaishnavi']

st.set_page_config(page_title = "CNN_Image_Classifier", layout = 'centered')

st.sidebar.title("Upload the image")
st.markdown("This application will try to give a classification of your image. It is build based on Valina CNN architecture ")

upload_file = st.sidebar.file_uploader("Chose your image", type = ['jpg', 'jpeg', 'png'])

from PIL import Image

if upload_file is not None:
    img = Image.open(upload_file).convert('RGB')
    st.image(img, caption = "Your Image")

    # Input image data preprocessing
    image_resized =img.resize((128,128)) # Resize the image
    img_array = image.img_to_array(image_resized)/255.0 # Image to array conversion
    image_batch = np.expand_dims(img_array, axis = 0) # Flattening the image data

    prediction = model.predict(image_batch)

    predicted_class = class_names[np.argmax(prediction)]

    st.success(f"This image is predicted to be: {predicted_class}")
   
    st.subheader("Below is yout confidance score for all the class")

    print(prediction)
    for index, score in enumerate(prediction[0]):
        st.write(f"{class_names[index]}: {score}")
    
