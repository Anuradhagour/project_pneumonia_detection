import streamlit as st
import numpy as np
import cv2
from PIL import Image
from keras.models import load_model
import tensorflow as tf

# Page Config
st.set_page_config(page_title="Pneumonia Detection", layout="centered")

st.title("🫁 X-ray Pneumonia Detection App")

#with st.expander("📘 Project Overview & Medical Background"):
st.markdown("""
    ### 🫁 Pneumonia Detection Using Chest X-ray

    Pneumonia is a lung infection that causes inflammation in the air sacs (alveoli).  
    It is a major health concern, especially in infants, elderly people, and those with weak immunity.

    ### 🔍 Why Chest X-ray?
    Chest X-ray is the most widely used imaging technique to diagnose pneumonia.
    Radiologists look for:
    - White/opacified regions in the lungs
    - Fluid or congestion in air spaces
    - Reduced clarity of lung structures

    ### 🤖 Why Deep Learning?
    Manual diagnosis can be:
    - Subjective (varies by radiologist)
    - Time-consuming in emergency cases

    Deep learning models like CNNs can:
    ✔ Automatically analyze X-rays  
    ✔ Detect abnormal patterns  
    ✔ Assist radiologists for faster decisions  

    ### 🎯 What This App Does
    - Upload a chest X-ray image
    - Model predicts whether the patient has **Pneumonia or Normal lungs**
    - If pneumonia detected → bounding boxes highlight abnormal regions*
    
    _*Bounding boxes are estimated using image processing since classification models do not directly output lesion coordinates._
    """)

#st.info("➡ Go to **Prediction Page** from the left sidebar to upload an X-ray.")


st.title("🩻 Pneumonia Prediction From X-ray")
# Load Model
@st.cache_resource
def load_pneumonia_model():
    model = load_model("best_stage1_finetuned.keras")
    return model

model = load_pneumonia_model()

# Image Preprocessing
def preprocess_image(img):
    img = img.convert("L")  # Convert to grayscale
    img = img.resize((224, 224))
    arr = np.array(img).astype("float32") / 255.0
    arr = arr.reshape(1, 224, 224, 1)
    return arr


# Bounding Box Detection (based on infection opacity)
def draw_bounding_boxes(original_img):
    img = np.array(original_img.convert("L"))
    img = cv2.resize(img, (224, 224))

    # Threshold based on lung opacity (bright area = infection)
    _, thresh = cv2.threshold(img, 120, 255, cv2.THRESH_BINARY)

    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL,
                                   cv2.CHAIN_APPROX_SIMPLE)

    bbox_img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)

    for cnt in contours:
        x, y, w, h = cv2.boundingRect(cnt)
        if w * h > 300:  # filter out noise
            cv2.rectangle(bbox_img, (x, y),
                          (x + w, y + h), (0, 0, 255), 2)

    return bbox_img


# Upload Section
uploaded_file = st.file_uploader("Upload Chest X-ray Image", type=["png", "jpg", "jpeg"])

if uploaded_file:
    img = Image.open(uploaded_file)
    st.image(img, caption="Uploaded X-ray", width=350)

    img_arr = preprocess_image(img)

    prediction = float(model.predict(img_arr)[0][0])
    Target = "1" if prediction > 0.5 else "Normal"
    confidence = prediction * 100 if prediction > 0.5 else (1 - prediction) * 100
    st.subheader(f"Prediction: **{ Target}**")
    st.write(f"Confidence: **{confidence:.2f}%**")

    if Target == "1":
        st.write("📌 Pneumonia region highlighted below:")
        detected_img = draw_bounding_boxes(img)
        st.image(detected_img, caption="Detected Infection Area", width=350)

