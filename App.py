%%writefile app.py
import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image

# --- CONFIGURATION ---
MODEL_PATH = '/content/tb_model.keras'
IMG_HEIGHT = 224
IMG_WIDTH = 224

# --- MODEL LOADING ---
@st.cache_resource
def load_trained_model():
    try:
        model = tf.keras.models.load_model(MODEL_PATH)
        return model
    except Exception as e:
        st.error(f"Error loading model: {e}. Please ensure 'tb_model.keras' is uploaded.")
        return None

model = load_trained_model()

# --- HELPER FUNCTION ---
def preprocess_image(image):
    image = image.resize((IMG_HEIGHT, IMG_WIDTH))
    if image.mode != 'RGB':
        image = image.convert('RGB')
    img_array = np.array(image)
    img_array_expanded = np.expand_dims(img_array, axis=0)
    return img_array_expanded / 255.0

# --- STREAMLIT APP LAYOUT ---
st.title('Tuberculosis Detection from Chest X-Rays 🩺')
st.markdown("Upload a chest X-ray image for prediction.")

uploaded_file = st.file_uploader("Choose an X-ray image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None and model is not None:
    image = Image.open(uploaded_file)
    st.image(image, caption='Uploaded X-Ray', use_container_width=True)

    with st.spinner('Analyzing the image...'):
        processed_image = preprocess_image(image)
        prediction = model.predict(processed_image)
        score = prediction[0][0]

        st.subheader("Prediction Result")
        if score > 0.5:
            st.warning(f"**Result: Tuberculosis**")
        else:
            st.success(f"**Result: Normal**")

        confidence = score * 100 if score > 0.5 else (1 - score) * 100
        st.write(f"Confidence: **{confidence:.2f}%**")
elif model is None:
    st.stop()
