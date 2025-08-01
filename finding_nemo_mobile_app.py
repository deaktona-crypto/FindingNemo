import streamlit as st
from PIL import Image
import folium
from streamlit_folium import st_folium
from ultralytics import YOLO
import tempfile
import os

# -----------------------------
# Page Config
# -----------------------------
st.set_page_config(page_title="ReefLog – FindingNemo", page_icon="🌊", layout="centered")
st.title("🐠 FindingNemo – Protect Marine Life")
st.write("Upload a photo of marine life or pollution, and our AI will try to identify it!")

# -----------------------------
# Load YOLOv8 model (pre-trained)
# -----------------------------
@st.cache_resource
def load_model():
    # You can replace yolov8n.pt with your custom-trained model (marine_best.pt)
    return YOLO("best.pt")

model = load_model()

# -----------------------------
# Image Upload & Detection
# -----------------------------
uploaded_file = st.file_uploader("📷 Upload an image", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    image = Image.open(uploaded_file)
    st.image(image, caption="Uploaded Image", use_container_width=True)
    st.write("⏳ Analyzing image with AI model...")

    # Save image temporarily
    with tempfile.NamedTemporaryFile(delete=False, suffix=".jpg") as temp_file:
        image.save(temp_file.name)
        temp_path = temp_file.name

    # Run YOLO detection
    results = model(temp_path)

    # Get detection info
    labels = results[0].names
    detected_items = []
    for box in results[0].boxes:
        cls_id = int(box.cls)
        conf = float(box.conf)
        detected_items.append(f"{labels[cls_id]} ({conf*100:.1f}%)")

    if detected_items:
        st.success("✅ Detected: " + ", ".join(detected_items))
    else:
        st.warning(⚠️ No objects detected.")

    # Show detection image
    st.image(results[0].plot(), caption="Detection Results", use_container_width=True)

    os.unlink(temp_path)  # Clean up temporary file

# -----------------------------
# Map Section
# -----------------------------
st.subheader("🌍 Global Reports")

map_center = [41.3275, 19.8189]  # Centered on Albania
m = folium.Map(location=map_center, zoom_start=6)

# Example pollution report marker
folium.Marker(
    [41.3275, 19.8189],
    popup="Plastic pollution near coast",
    tooltip="Click for more info"
).add_to(m)

st_folium(m, width=700, height=450)

# -----------------------------
# Footer
# -----------------------------
st.markdown("---")
st.caption("🐟 FindingNemo – Hackathon Prototype with YOLOv8 AI Detection")
