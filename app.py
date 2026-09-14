import streamlit as st
import cv2
import tempfile
from detector import detect_trees

st.set_page_config(
    page_title="ForestLens",
    page_icon="🌲",
    layout="wide"
)

# ---------- CUSTOM CSS ----------
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Poppins:wght@300;400;600;700&display=swap');

html, body, [class*="css"]{
    font-family: 'Poppins', sans-serif;
    background: #07111A;
}

.main{
    background: linear-gradient(180deg,#07111A 0%, #0B1F16 100%);
}

.hero{
    padding:35px;
    border-radius:22px;
    background: linear-gradient(135deg,#0F3D2E,#14532D);
    color:white;
    margin-bottom:20px;
}

.hero h1{
    font-size:48px;
    margin:0;
}

.hero p{
    opacity:0.9;
    font-size:18px;
}

.card{
    background: rgba(255,255,255,0.06);
    backdrop-filter: blur(10px);
    border:1px solid rgba(255,255,255,0.08);
    border-radius:18px;
    padding:20px;
    text-align:center;
}

.metric{
    font-size:34px;
    color:#4ADE80;
    font-weight:700;
}

.label{
    color:#C7D2FE;
    font-size:14px;
}

.section{
    background:#0B1720;
    border-radius:18px;
    padding:18px;
    border:1px solid #1F2937;
}

.footer{
    text-align:center;
    color:#94A3B8;
    padding:30px;
}
</style>
""", unsafe_allow_html=True)

# ---------- HERO ----------
st.markdown("""
<div class="hero">
    <h1>🌲 ForestLens</h1>
    <p>AI-powered Forest Canopy & Tree Crown Detection using Deep Learning</p>
</div>
""", unsafe_allow_html=True)

# ---------- SIDEBAR ----------
with st.sidebar:
    st.sidebar.image("assets/logo.jpg", width=120)
    st.title("⚙️ Analysis")
    uploaded = st.file_uploader(
        "Upload Satellite Image",
        type=["jpg","jpeg","png","tif","tiff"]
    )

    st.markdown("---")
    st.info("Supported: RGB aerial & satellite imagery")

# ---------- MAIN ----------
if uploaded:

    tmp = tempfile.NamedTemporaryFile(delete=False)
    tmp.write(uploaded.read())

    with st.spinner("🛰️ AI is analyzing the forest canopy..."):
        image, data, count, canopy_pixels, canopy_percent, conf = detect_trees(tmp.name)

    # Metrics
    c1,c2,c3,c4 = st.columns(4)

    with c1:
        st.markdown(f"""
        <div class="card">
            <div class="metric">{count}</div>
            <div class="label">Trees Detected</div>
        </div>
        """, unsafe_allow_html=True)

    with c2:
        st.markdown(f"""
        <div class="card">
            <div class="metric">{canopy_percent:.1f}%</div>
            <div class="label">Canopy Coverage</div>
        </div>
        """, unsafe_allow_html=True)

    with c3:
        st.markdown(f"""
        <div class="card">
            <div class="metric">{conf:.2f}</div>
            <div class="label">AI Confidence</div>
        </div>
        """, unsafe_allow_html=True)

    with c4:
        st.markdown(f"""
        <div class="card">
            <div class="metric">{canopy_pixels:,}</div>
            <div class="label">Crown Pixels</div>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    # Images
    left,right = st.columns([1,1])

    with left:
        st.markdown("### 📸 Original Image")
        st.image(uploaded, use_container_width=True)

    with right:
        st.markdown("### 🌳 AI Detection")
        rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        st.image(rgb, use_container_width=True)

    st.markdown("<br>", unsafe_allow_html=True)

    # Detection table
    st.markdown('<div class="section">', unsafe_allow_html=True)
    st.subheader("📋 Detection Results")
    st.dataframe(data, use_container_width=True)

    csv = data.to_csv(index=False).encode("utf-8")

    st.download_button(
        "⬇ Download CSV Report",
        csv,
        "forestlens_results.csv",
        "text/csv"
    )
    st.markdown("</div>", unsafe_allow_html=True)

else:

    st.markdown("""
    <div class="section" style="text-align:center;padding:60px;">
        <h2>📤 Upload a Satellite Image</h2>
        <p>Start by uploading an aerial RGB image to detect individual tree crowns.</p>
    </div>
    """, unsafe_allow_html=True)

# ---------- FOOTER ----------
st.markdown("""
<div class="footer">
ForestLens • AI for Sustainable Forest Monitoring 🌍
</div>
""", unsafe_allow_html=True)