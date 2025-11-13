import streamlit as st
from ultralytics import YOLO
import numpy as np
from PIL import Image
import cv2
import matplotlib.pyplot as plt
import os
from pathlib import Path

st.set_page_config(
    page_title="Medical Image Segmentation",
    layout="wide"
)

st.markdown(
    """
    <style>
      .block-container {max-width: 1200px; padding-top: 1rem;}
      header, footer {visibility: hidden;}
      .stProgress > div > div > div > div { transition: width 0.6s ease; }
      .caption-muted { color: #6b7280; }
      [data-testid="stImage"] img {
        max-height: 520px;
        width: auto !important;
        object-fit: contain;
      }
    </style>
    """,
    unsafe_allow_html=True
)


@st.cache_resource
def load_model():
    model = YOLO('best_1.pt')
    return model

def preprocess_image(image_path, target_size=(640, 640)):
    img = cv2.imread(image_path)
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img_display = img_rgb.copy()
    return img_rgb, img_display

def predict_image(model, image_path, conf=0.5, iou=0.45, imgsz=640):
    results = model.predict(
        source=image_path,
        conf=conf,
        iou=iou,
        imgsz=imgsz,
        save=False,
        verbose=False
    )
    return results[0]

def draw_boxes(image, result, model):
    annotated_image = image.copy()
    
    if len(result.boxes) > 0:
        for box in result.boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            cls_id = int(box.cls[0])
            confidence = float(box.conf[0])
            
            color = (0, 255, 0)
            thickness = 3
            cv2.rectangle(annotated_image, (x1, y1), (x2, y2), color, thickness)
            
            label = f"{model.names[cls_id]} {confidence:.2%}"
            font_scale = 1.2
            font_thickness = 2
            
            text_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, font_scale, font_thickness)[0]
            text_x = x1
            text_y = y1 - 10
            
            cv2.rectangle(
                annotated_image,
                (text_x, text_y - text_size[1] - 8),
                (text_x + text_size[0] + 6, text_y + 8),
                color,
                -1
            )
            
            cv2.putText(
                annotated_image,
                label,
                (text_x + 3, text_y),
                cv2.FONT_HERSHEY_SIMPLEX,
                font_scale,
                (0, 0, 0),
                font_thickness
            )
    
    return annotated_image

def main():
    st.title("Medical Image Segmentation")
    st.markdown(
        "<p class='caption-muted'>Upload or select a medical image to obtain model predictions with segmentation boxes.</p>",
        unsafe_allow_html=True
    )

    st.sidebar.header("Configuration")
    
    try:
        model = load_model()
    except Exception as e:
        st.sidebar.error(f"Error loading model: {str(e)}")
        st.stop()
    
    threshold = st.sidebar.slider(
        "Confidence threshold",
        min_value=0.0, max_value=1.0, value=0.5, step=0.05,
        help="Minimum probability required for a detection to be displayed."
    )
    
    iou_threshold = st.sidebar.slider(
        "IOU threshold",
        min_value=0.0, max_value=1.0, value=0.45, step=0.05,
        help="Non-maximum suppression threshold."
    )
    
    img_size = st.sidebar.selectbox(
        "Image size",
        options=[384, 512, 640],
        index=2,
        help="Model input image size."
    )
    
    image_path = None
    
    uploaded_file = st.file_uploader(
        "Upload medical image",
        type=['png', 'jpg', 'jpeg'],
        help="Supported formats: PNG, JPG, JPEG."
    )
    
    if uploaded_file is not None:
        with open("temp_image.png", "wb") as f:
            f.write(uploaded_file.getbuffer())
        image_path = "temp_image.png"

    if image_path and os.path.exists(image_path):
        img_rgb, img_display = preprocess_image(image_path)
        
        tab_img, tab_results = st.tabs(["Image", "Results"])

        with st.spinner("Analyzing image..."):
            result = predict_image(model, image_path, conf=threshold, iou=iou_threshold, imgsz=img_size)
            annotated_img = draw_boxes(img_display, result, model)

        with tab_img:
            st.subheader("Input image")
            st.image(
                img_display,
                channels="RGB",
                caption="Original image",
                use_column_width=False,
                width=720
            )

        with tab_results:
            st.subheader("Analysis results")
            
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Objects Detected", len(result.boxes))
            with col2:
                if len(result.boxes) > 0:
                    avg_conf = np.mean([float(box.conf[0]) for box in result.boxes])
                    st.metric("Average Confidence", f"{avg_conf:.2%}")
                else:
                    st.metric("Average Confidence", "N/A")
            with col3:
                image_name = Path(image_path).stem
                st.metric("Image", image_name[:25])
            
            st.markdown("---")
            
            detections_data = []
            for box in result.boxes:
                cls_id = int(box.cls[0])
                confidence = float(box.conf[0])
                detections_data.append({
                    "Class": model.names[cls_id],
                    "Confidence": confidence
                })
            
            detections_data.sort(key=lambda x: x["Confidence"], reverse=True)
            
            st.markdown("**All Label Probabilities**")
            if len(detections_data) > 0:
                for detection in detections_data:
                    st.progress(
                        detection["Confidence"],
                        text=f"{detection['Class']} — {detection['Confidence']:.4f} ({detection['Confidence']*100:.2f}%)"
                    )
            else:
                st.write("No detections found.")
            
            st.markdown("---")
            
            st.markdown(f"**Predictions Above Threshold (≥ {threshold:.2f})**")
            above_threshold = [d for d in detections_data if d["Confidence"] >= threshold]
            
            if len(above_threshold) > 0:
                st.write(f"Found **{len(above_threshold)}** object(s) above threshold:")
                for i, detection in enumerate(above_threshold, 1):
                    st.write(f"- **{detection['Class']}**: {detection['Confidence']*100:.2f}%")
            else:
                st.write("No objects detected above the current threshold.")
            
            st.markdown("---")
            st.subheader("Segmentation with Bounding Boxes")
            st.image(
                annotated_img,
                channels="RGB",
                caption="Image with detected objects and bounding boxes",
                use_column_width=True
            )
            
            st.caption("This tool reports model outputs and does not replace clinical judgment.")
    else:
        st.info("Please upload a medical image to begin.")
        
        with st.expander("Instructions"):
            st.markdown("""
            1. Upload a medical image (PNG/JPG/JPEG)
            2. Adjust confidence and IOU thresholds in the sidebar if needed
            3. Review detection results and segmentation boxes
            4. Analyze confidence scores for each detected object
            """)

if __name__ == "__main__":
    main()