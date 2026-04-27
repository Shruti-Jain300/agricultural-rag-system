import numpy as np
import PIL.Image as Image
import tensorflow as tf
import streamlit as st
from streamlit_extras.add_vertical_space import add_vertical_space
from warnings import filterwarnings
filterwarnings('ignore')


# ------------------ Streamlit Configuration ------------------
def streamlit_config():
    st.set_page_config(page_title='Plant Disease Classification', layout='centered')

    # Transparent header
    page_background_color = """
    <style>
    [data-testid="stHeader"] {
        background: rgba(0,0,0,0);
    }
    </style>
    """
    st.markdown(page_background_color, unsafe_allow_html=True)

    # Title
    st.markdown(
        '<h1 style="text-align: center;">🌿 Plant Disease Classification</h1>',
        unsafe_allow_html=True,
    )
    add_vertical_space(2)
    st.markdown(
        '<h4 style="text-align: center; color: gray;">Detect diseases in Potato and Tomato leaves using Deep Learning</h4>',
        unsafe_allow_html=True,
    )
    add_vertical_space(3)


# Call configuration
streamlit_config()


# ------------------ Prediction Function ------------------
def prediction(image_path, crop_type):
    # Load image
    img = Image.open(image_path)
    img_resized = img.resize((256, 256))
    img_array = tf.keras.preprocessing.image.img_to_array(img_resized)
    img_array = np.expand_dims(img_array, axis=0)

    # Choose model and class names
    if crop_type == "Potato":
        model_path = r"models/potato_classification_model.h5"
        class_names = ['Potato___Early_blight', 'Potato___Late_blight', 'Potato___healthy']
    elif crop_type == "Tomato":
        model_path = r"models/tomato_classification_model.h5"
        class_names = ['Tomato___Early_blight', 'Tomato___Late_blight', 'Tomato___healthy']
    else:
        st.error("Invalid crop type selected.")
        return

    # Load the selected model
    model = tf.keras.models.load_model(model_path)
    prediction = model.predict(img_array)

    # Get predicted class and confidence
    predicted_class = class_names[np.argmax(prediction)]
    confidence = round(np.max(prediction) * 100, 2)

    # Display results
    add_vertical_space(1)
    st.markdown(
        f'<h4 style="color: orange;">Predicted Class : {predicted_class}<br>Confidence : {confidence}%</h4>',
        unsafe_allow_html=True,
    )
    add_vertical_space(1)
    st.image(img.resize((400, 300)))


# ------------------ Streamlit UI Layout ------------------
col1, col2, col3 = st.columns([0.15, 0.7, 0.15])
with col2:
    # Select Crop Type
    crop_type = st.selectbox("Select Crop Type", ["Potato", "Tomato"])
    add_vertical_space(2)

    # File uploader
    input_image = st.file_uploader("Upload the Leaf Image", type=["jpg", "jpeg", "png"])

# ------------------ Run Prediction ------------------
if input_image is not None:
    col1, col2, col3 = st.columns([0.2, 0.8, 0.2])
    with col2:
        prediction(input_image, crop_type)

