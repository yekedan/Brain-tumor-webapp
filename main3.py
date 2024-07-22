import streamlit as st
from PIL import Image
import torch
from model import load_model, predict, get_writeup

# Load the model
model = load_model('brain_tumor_model.pth')

# Streamlit App
st.title("Brain Tumor Detection")

uploaded_file = st.file_uploader("Choose an image...", type="jpg")

if uploaded_file is not None:
    image = Image.open(uploaded_file)
    st.image(image, caption='Uploaded Image.', use_column_width=True)
    st.write("")
    st.write("Classifying...")
    
    label, confidence = predict(model, uploaded_file)
    writeup = get_writeup(label)
    
    st.write(f"Confidence: {confidence:.2f}")
    st.write(f"English: {writeup['en']}")
    st.write(f"French: {writeup['fr']}")
