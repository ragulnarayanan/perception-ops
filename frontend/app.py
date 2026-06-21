import streamlit as st
import requests
from PIL import Image
import io
import pandas as pd
from streamlit_webrtc import webrtc_streamer
import av
from ultralytics import YOLO

BASE_URL = os.getenv(
    "API_BASE_URL",
    "http://localhost:8000"
)

API_URL = f"{BASE_URL}/predict"

IMAGE_API_URL = (
    f"{BASE_URL}/predict-image"
)

VIDEO_API_URL = (
    f"{BASE_URL}/predict-video"
)

st.set_page_config(
    page_title="PerceptionOps",
    page_icon="🚗",
    layout="wide"
)

st.title("🚗 PerceptionOps")
st.subheader("Traffic Perception Platform")

tab1, tab2, tab3, tab4 = st.tabs(
    [
        "Image Detection",
        "Video Detection",
        "Live Camera",
        "Monitoring"
    ]
)

webcam_model = YOLO("models/best.pt")

class VideoProcessor:

    def recv(self, frame):

        img = frame.to_ndarray(
            format="bgr24"
        )

        results = webcam_model(
            img,
            conf=0.4
        )

        annotated = results[0].plot()

        return av.VideoFrame.from_ndarray(
            annotated,
            format="bgr24"
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
            if (
                image_response.status_code == 200
                and image_response.headers.get("content-type", "").startswith("image/")
            ):

                annotated_image = Image.open(
                    io.BytesIO(image_response.content)
                )

            else:

                st.error(
                    f"Image API failed: {image_response.text}"
                )

                st.stop()

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
        type=["mp4"],
        key="video_upload"
    )

    if video_file:

        st.subheader("Input Video")

        st.video(video_file)

        files = {
            "file": (
                video_file.name,
                video_file.getvalue(),
                video_file.type
            )
        }

        if st.button(
            "Run Video Detection"
        ):

            with st.spinner(
                "Processing Video..."
            ):

                response = requests.post(
                    VIDEO_API_URL,
                    files=files
                )

            if response.status_code == 200:

                output_path = "processed_video.mp4"

                with open(
                    output_path,
                    "wb"
                ) as f:
                    f.write(response.content)

                st.success(
                    "Video Processed"
                )

                st.subheader(
                    "Processed Video"
                )

                st.video(
                    output_path
                )

            else:

                st.error(
                    f"Video processing failed: {response.text}"
                )


with tab3:

    st.subheader(
        "Live Webcam Detection"
    )

    webrtc_streamer(
        key="traffic-camera",
        video_processor_factory=VideoProcessor
    )
# =====================================================
# MONITORING
# =====================================================

with tab4:

    st.subheader("System Monitoring Dashboard")

    st.components.v1.iframe(
        "http://localhost:3000/d/adfl4l8/preceptionops?orgId=1&kiosk=true",
        height=900,
        scrolling=True
)