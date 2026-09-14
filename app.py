import streamlit as st
import cv2
import tempfile
import pandas as pd
import numpy as np
from detector import detect_trees_hybrid
from PIL import Image
from detector import detect_trees_hybrid
st.set_page_config(
    page_title="ForestLens • AI Canopy Intelligence",
    page_icon="🌲",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ----------------- MODERN DARK GLASSMORPHISM STYLING -----------------
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@300;400;500;600;700;800&display=swap');

* {
    font-family: 'Plus Jakarta Sans', sans-serif;
}

/* Background gradient */
.stApp {
    background: radial-gradient(circle at 10% 20%, #0c1c14 0%, #060b08 100%);
    color: #e2e8f0;
}

/* Header Banner */
.hero-box {
    background: linear-gradient(135deg, rgba(16, 185, 129, 0.15) 0%, rgba(5, 150, 105, 0.05) 100%);
    border: 1px solid rgba(52, 211, 153, 0.25);
    border-radius: 20px;
    padding: 30px;
    margin-bottom: 25px;
    backdrop-filter: blur(12px);
    box-shadow: 0 10px 30px -10px rgba(0,0,0,0.5);
}

.hero-box h1 {
    font-size: 2.4rem;
    font-weight: 800;
    margin: 0;
    background: linear-gradient(90deg, #34d399, #10b981, #6ee7b7);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
}

.hero-box p {
    color: #94a3b8;
    margin-top: 6px;
    font-size: 1rem;
}

/* Modern Metric Cards */
.metric-card {
    background: rgba(18, 32, 25, 0.55);
    border: 1px solid rgba(52, 211, 153, 0.15);
    border-radius: 16px;
    padding: 20px;
    text-align: center;
    backdrop-filter: blur(10px);
    transition: all 0.3s cubic-bezier(0.4, 0, 0.2, 1);
    box-shadow: 0 4px 20px rgba(0, 0, 0, 0.3);
}

.metric-card:hover {
    transform: translateY(-4px);
    border-color: rgba(52, 211, 153, 0.45);
    box-shadow: 0 12px 28px rgba(16, 185, 129, 0.15);
}

.metric-val {
    font-size: 2.2rem;
    font-weight: 800;
    color: #34d399;
    letter-spacing: -0.5px;
}

.metric-title {
    font-size: 0.85rem;
    font-weight: 600;
    text-transform: uppercase;
    letter-spacing: 1px;
    color: #94a3b8;
    margin-top: 4px;
}

/* Glass Section Container */
.glass-container {
    background: rgba(15, 23, 19, 0.6);
    border: 1px solid rgba(255, 255, 255, 0.07);
    border-radius: 18px;
    padding: 24px;
    margin-top: 20px;
    backdrop-filter: blur(8px);
}
</style>
""", unsafe_allow_html=True)

# ----------------- SIDEBAR CONTROLS -----------------
with st.sidebar:
    st.image("assets/logo.jpg", width=90)
    st.markdown("### **ForestLens Studio**")
    st.caption("AI-Powered Forest Canopy Analytics")
    st.markdown("---")
    
    uploaded = st.file_uploader(
        "Upload Aerial / Satellite Image",
        type=["jpg", "jpeg", "png", "tif", "tiff"],
        help="Upload high-res drone or satellite imagery (RGB)"
    )
    
    st.markdown("---")
    st.markdown("#### **Detection Settings**")
    conf_threshold = st.slider("Confidence Filter", 0.10, 0.90, 0.25, 0.05)
    
    st.info("💡 **Tip:** Adjust confidence threshold to filter out ambiguous canopy shadows.")
with st.sidebar:
    st.markdown("### Detection Settings")
    
    detection_mode = st.radio(
        "Detection Algorithm",
        options=["Standard DL Model", "High-Density / Hybrid (Optimistic)"],
        index=1,
        help="Use Hybrid to capture overlapping understory and shaded lower crowns."
    )
    
    if detection_mode == "High-Density / Hybrid (Optimistic)":
        min_distance = st.slider(
            "Crown Spacing (px)", 
            min_value=6, 
            max_value=30, 
            value=12,
            help="Lower values detect smaller/tighter understory crowns."
        )
        green_thresh = st.slider("Canopy Sensitivity", 20, 80, 45)
    else:
        conf_filter = st.slider("Confidence Filter", 0.10, 0.90, 0.30)
# --- Execution & Metric Display ---
uploaded_file = st.sidebar.file_uploader("Upload Aerial / Satellite Image", type=["jpg", "png", "jpeg"])

if uploaded_file:
    input_img = Image.open(uploaded_file).convert("RGB")
    
    if detection_mode == "High-Density / Hybrid (Optimistic)":
        results = detect_trees_hybrid(input_img, min_peak_distance=min_distance, green_thresh=green_thresh)
        tree_count = results["tree_count"]
        density = results["canopy_density"]
        crown_pixels = results["crown_pixels"]
        annotated_img = results["annotated_image"]
        mean_conf = "N/A (Density)"
    else:
        # Call your existing DL model inference function here
        pass

    # Top KPI Metrics
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("TREES DETECTED", tree_count)
    col2.metric("CANOPY DENSITY", f"{density}%")
    col3.metric("MEAN CONFIDENCE", str(mean_conf))
    col4.metric("CROWN PIXELS", f"{crown_pixels:,}")

    # Display Side-by-Side Images
    c_orig, c_pred = st.columns(2)
    with c_orig:
        st.subheader("📷 Original Image")
        st.image(input_img, use_container_width=True)
    with c_pred:
        st.subheader("🌲 Model Predictions (Bounding Boxes)")
        st.image(annotated_img, use_container_width=True)

# ----------------- MAIN VIEW -----------------
st.markdown("""
<div class="hero-box">
    <h1>🌲 ForestLens AI</h1>
    <p>Autonomous Canopy Segmentation & Individual Tree Crown (ITC) Detection</p>
</div>
""", unsafe_allow_html=True)

if uploaded:
    with tempfile.NamedTemporaryFile(delete=False) as tmp:
        tmp.write(uploaded.read())
        tmp_path = tmp.name

    with st.spinner("🛰️ DeepForest neural model analyzing tree crowns..."):
        image = Image.open(tmp_path).convert("RGB")
        data, count, canopy_pixels, canopy_percent, conf = detect_trees_hybrid(image)

    # Filter data based on sidebar threshold slider
    if not data.empty and 'score' in data.columns:
        filtered_data = data[data['score'] >= conf_threshold]
        count = len(filtered_data)
        conf = float(filtered_data['score'].mean()) if count > 0 else 0.0

    # 4 Glassmorphism Metric Cards
    c1, c2, c3, c4 = st.columns(4)
    with c1:
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-val">{count:,}</div>
            <div class="metric-title">Trees Detected</div>
        </div>
        """, unsafe_allow_html=True)
    with c2:
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-val">{canopy_percent:.1f}%</div>
            <div class="metric-title">Canopy Density</div>
        </div>
        """, unsafe_allow_html=True)
    with c3:
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-val">{conf:.2f}</div>
            <div class="metric-title">Mean Confidence</div>
        </div>
        """, unsafe_allow_html=True)
    with c4:
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-val">{canopy_pixels:,}</div>
            <div class="metric-title">Crown Pixels</div>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    # Visual Inspection Section with Tabs
    st.markdown("### 🔍 Visual Analysis")
    tab_side_by_side, tab_overlay, tab_analytics = st.tabs(["⚡ Side-by-Side View", "🎯 Overlay Only", "📊 Model Distribution"])

    rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    with tab_side_by_side:
        col_left, col_right = st.columns(2)
        with col_left:
            st.markdown("##### 📷 Original Image")
            st.image(uploaded, use_container_width=True)
        with col_right:
            st.markdown("##### 🌲 Model Predictions (Bounding Boxes)")
            st.image(rgb, use_container_width=True)

    with tab_overlay:
        st.image(rgb, caption="High-Resolution Detection Map", use_container_width=True)

    with tab_analytics:
        if not data.empty and 'score' in data.columns:
            st.markdown("##### 📈 Prediction Score Distribution")
            hist_vals, bin_edges = np.histogram(data['score'], bins=20, range=(0.1, 1.0))
            chart_data = pd.DataFrame({'Confidence Score': bin_edges[:-1], 'Tree Count': hist_vals})
            st.bar_chart(chart_data.set_index('Confidence Score'), color="#10b981")

    # Data Table & Export
    with st.expander("📋 View Raw Detection Geo-Data & Download Report", expanded=True):
        st.dataframe(data, use_container_width=True, height=260)
        csv = data.to_csv(index=False).encode("utf-8")
        st.download_button(
            label="⬇ Download Detection CSV",
            data=csv,
            file_name="forestlens_tree_inventory.csv",
            mime="text/csv",
            help="Download the detection coordinates and confidence scores"
        )

else:
    st.markdown("""
    <div style="border: 2px dashed rgba(52, 211, 153, 0.3); border-radius: 18px; padding: 60px; text-align: center; margin-top: 30px;">
        <h3 style="color: #6ee7b7;">Ready for Forest Analysis</h3>
        <p style="color: #94a3b8; max-width: 500px; margin: 0 auto;">
            Upload an aerial or satellite RGB image using the left sidebar to generate automated tree counts, canopy cover percentages, and crown metrics.
        </p>
    </div>
    """, unsafe_allow_html=True)