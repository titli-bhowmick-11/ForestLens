import streamlit as st
import cv2
import tempfile
import pandas as pd
from detector import detect_trees
from geospatial import read_kml

st.set_page_config(page_title="ForestLens", layout="wide")
st.title("🌳 ForestLens")
st.caption("AI Forest Canopy Analyzer")

# 1. Image Uploader
uploaded = st.file_uploader(
    "Upload aerial / satellite image", 
    type=["png", "jpg", "jpeg", "tif", "tiff"]
)

# 2. Optional KML Boundary Uploader
kml_file = st.file_uploader(
    "Upload KML boundary (optional)", 
    type=["kml"]
)

# Process KML if uploaded
if kml_file:
    with tempfile.NamedTemporaryFile(delete=False, suffix=".kml") as tmp_kml:
        tmp_kml.write(kml_file.read())
        kml_path = tmp_kml.name
    
    area_ha = read_kml(kml_path)
    st.metric("Forest Area", f"{area_ha:.2f} ha")

# Process Image if uploaded
if uploaded:
    with tempfile.NamedTemporaryFile(delete=False) as tmp:
        tmp.write(uploaded.read())
        tmp_path = tmp.name

    with st.spinner("Analyzing forest..."):
        image, data, count, canopy_pixels, canopy_percent, conf = detect_trees(tmp_path)

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("🌳 Trees", count)
    c2.metric("🌿 Canopy %", f"{canopy_percent:.1f}%")
    c3.metric("🎯 Confidence", f"{conf:.2f}")
    c4.metric("📦 Crown Pixels", f"{canopy_pixels:,}")

    rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    st.image(rgb, use_container_width=True)

    st.subheader("Detection Data")
    st.dataframe(data)

    csv = data.to_csv(index=False).encode("utf-8")
    st.download_button("⬇ Download CSV", csv, "tree_detection.csv", "text/csv")

    st.divider()
    st.info(
        """
        **Disclaimer**

        This tool estimates tree crowns from RGB aerial imagery using a pretrained AI model.

        Accuracy depends on image resolution, shadows, overlapping canopies, and image quality.
        """
    )