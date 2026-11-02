import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image
from pathlib import Path

st.set_page_config(page_title="Waste Classification", page_icon="♻️")
st.title("♻️ Waste Classification System")
st.write("Upload a clear image. The trained CNN model will predict one of its six learned categories.")

MODEL_FILE = Path("waste_classifier.keras")
LABEL_FILE = Path("class_names.txt")

@st.cache_resource
def load_model():
    return tf.keras.models.load_model(MODEL_FILE)

if not MODEL_FILE.exists() or not LABEL_FILE.exists():
    st.error("Model files are missing. Add waste_classifier.keras and class_names.txt beside app.py.")
    st.stop()

model = load_model()
class_names = LABEL_FILE.read_text(encoding="utf-8").splitlines()

uploaded = st.file_uploader("Choose a waste image", type=["jpg", "jpeg", "png"])

if uploaded:
    image = Image.open(uploaded).convert("RGB")
    st.image(image, caption="Uploaded image", use_container_width=True)

    if st.button("Predict"):
        resized = image.resize((128, 128))
        array = np.asarray(resized, dtype=np.float32)
        batch = np.expand_dims(array, axis=0)
        probabilities = model.predict(batch, verbose=0)[0]
        index = int(np.argmax(probabilities))
        st.subheader(f"Predicted category: {class_names[index].title()}")
        st.metric("Model confidence", f"{probabilities[index] * 100:.1f}%")
        st.caption("Confidence is the model's score, not a guarantee that the prediction is correct.")
