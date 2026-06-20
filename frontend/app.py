import streamlit as st
import requests
from PIL import Image
import io
import pandas as pd

API_URL = "http://localhost:8000/predict"
IMAGE_API_URL = "http://localhost:8000/predict-image"
VIDEO_API_URL = "http://localhost:8000/predict-video"

st.set_page_config(
    page_title="PerceptionOps",
    page_icon="🚗",
    layout="wide"
)

st.title("🚗 PerceptionOps")
st.subheader("Traffic Perception Platform")

tab1, tab2, tab3 = st.tabs(
    [
        "Image Detection",
        "Video Detection",
        "Monitoring"
    ]
)

# =====================================================
# IMAGE DETECTION
# =====================================================

with tab1:

    uploaded_file = st.file_uploader(
        "Upload Image",
        type=["jpg", "jpeg", "png"]
    )

    if uploaded_file:

        image = Image.open(uploaded_file)

        files = {
            "file": (
                uploaded_file.name,
                uploaded_file.getvalue(),
                uploaded_file.type
            )
        }

        with st.spinner("Running inference..."):

            response = requests.post(
                API_URL,
                files=files
            )

            result = response.json()

            image_response = requests.post(
                IMAGE_API_URL,
                files=files
            )

            annotated_image = Image.open(
                io.BytesIO(image_response.content)
            )

        col1, col2 = st.columns(2)

        with col1:
            st.image(
                image,
                caption="Input Image",
                use_container_width=True
            )

        with col2:
            st.image(
                annotated_image,
                caption="Predictions",
                use_container_width=True
            )

        m1, m2, m3 = st.columns(3)

        m1.metric(
            "Latency (ms)",
            result["latency_ms"]
        )

        m2.metric(
            "Detections",
            result["num_detections"]
        )

        m3.metric(
            "Model",
            result["model_version"]
        )

        st.subheader("Class Distribution")

        class_df = pd.DataFrame(
            list(result["class_distribution"].items()),
            columns=["Class", "Count"]
        )

        st.dataframe(
            class_df,
            use_container_width=True
        )

# =====================================================
# VIDEO DETECTION
# =====================================================

with tab2:

    video_file = st.file_uploader(
        "Upload Video",
        type=["mp4"]
    )

    if video_file:

        st.video(video_file)

        files = {
            "file": (
                video_file.name,
                video_file.getvalue(),
                video_file.type
            )
        }

        if st.button("Run Video Detection"):

            with st.spinner("Processing Video..."):

                response = requests.post(
                    VIDEO_API_URL,
                    files=files
                )

            st.success("Video Processed")

            st.video(
                response.content
            )

# =====================================================
# MONITORING
# =====================================================

with tab3:

    st.subheader("System Monitoring Dashboard")

    st.markdown(
        """
        <iframe
            src="http://localhost:3000"
            width="100%"
            height="900"
            frameborder="0">
        </iframe>
        """,
        unsafe_allow_html=True
    )