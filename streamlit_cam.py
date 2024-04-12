import streamlit as st
from transformers import DetrImageProcessor, DetrForObjectDetection
from PIL import Image, ImageDraw, ImageFont
from ultralytics import YOLO
from io import BytesIO
import cv2
import torch
import numpy as np
import tempfile
import time

# Authenticate to Google Cloud Storage if not already done
# Make sure to set up the authentication properly before running this code

# Define the weights folder
weights_folder = "weights/"

def annotate_image(image):
    try:
        # Load DETR model and processor for person detection
        detr_processor = DetrImageProcessor.from_pretrained("facebook/detr-resnet-50", revision="no_timm")
        detr_model = DetrForObjectDetection.from_pretrained("facebook/detr-resnet-50", revision="no_timm")

        # Process image for person detection
        detr_inputs = detr_processor(images=image, return_tensors="pt")
        detr_outputs = detr_model(**detr_inputs)

        # Convert outputs to COCO API for person detection
        detr_target_sizes = torch.tensor([image.size[::-1]])
        detr_results = detr_processor.post_process_object_detection(detr_outputs, target_sizes=detr_target_sizes, threshold=0.5)[0]

        # Check if a person is detected in the image
        person_detected = any(detr_model.config.id2label[label.item()] == "person" for label in detr_results["labels"])

        if person_detected:
            # Draw bounding boxes and labels for persons on the image
            detr_draw = ImageDraw.Draw(image)
            person_count = 0  # Counter for numbering persons
            person_boxes = {}  # Dictionary to store bounding boxes for each person
            for score, label, box in zip(detr_results["scores"], detr_results["labels"], detr_results["boxes"]):
                if detr_model.config.id2label[label.item()] == "person":
                    person_count += 1
                    box = [round(i, 2) for i in box.tolist()]
                    person_boxes[f"Person {person_count}"] = box

            # Load the trained YOLO model for product type detection
            product_weight_file_path = "./weights/best_obj_detect_prod_types.pt"  # Update with actual path
            product_model = YOLO(product_weight_file_path)

            # Load the trained YOLO model for logo detection
            logo_weight_file_path = "./weights/best_obj_detect_logos.pt"  # Update with actual path
            logo_model = YOLO(logo_weight_file_path)

            # Set the detection threshold (e.g., 0.3 for 30% confidence)
            logo_detection_threshold = 0.4

            # Draw bounding boxes and labels for persons on the image
            for person_id, box in person_boxes.items():
                x_min, y_min, x_max, y_max = box
                person_image = image.crop((x_min, y_min, x_max, y_max))  # Crop the image to the person's bounding box
                
                # Calculate dynamic label position
                label_x = x_min
                label_y = y_min - 20  # Initial position above the bounding box
                if label_y < 0:  # If label is out of bounds, place it below the bounding box
                    label_y = y_max + 5  # Adjust the value to your preference

                # Detect product types
                product_results = product_model.predict(person_image)
                product_labels = [product_model.names[int(obj.cls[0])] for obj in product_results[0].boxes]

                # Detect logos
                logo_results = logo_model.predict(person_image, conf=logo_detection_threshold)
                logo_labels = [logo_model.names[int(obj.cls[0])] for obj in logo_results[0].boxes]

                # Construct the label text
                label_text = ""
                unique_combinations = set()
                for product_label, logo_label in zip(product_labels, logo_labels):
                    combination = (logo_label, product_label)
                    if combination not in unique_combinations:
                        label_text += f"Wearing {logo_label} {product_label}\n"
                        unique_combinations.add(combination)
                if label_text:
                    # Draw the bounding box and label
                    detr_draw.rectangle(box, outline="blue", width=2)
                    detr_draw.text((label_x, label_y), label_text, font=ImageFont.truetype("arial.ttf", size=16), fill="blue")
                    
        else:
            st.warning("No person detected in the image.")
    except Exception as e:
        st.error(f"Error processing image: {e}")

def annotate_video(uploaded_video):
    try:
        # Save the uploaded video to a temporary file
        with tempfile.NamedTemporaryFile(delete=False, suffix=".mp4") as temp_file:
            temp_file.write(uploaded_video.read())
            temp_file_path = temp_file.name

        # Load the video
        video = cv2.VideoCapture(temp_file_path)

        # Get the video properties
        fps = video.get(cv2.CAP_PROP_FPS)
        width = int(video.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(video.get(cv2.CAP_PROP_FRAME_HEIGHT))

        # Take snapshots at 0 seconds and 2 seconds
        video.set(cv2.CAP_PROP_POS_MSEC, 0)
        ret, frame = video.read()
        if not ret:
            st.error("Error reading the video file.")
            return
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        image1 = Image.fromarray(frame)
        annotate_image(image1)

        video.set(cv2.CAP_PROP_POS_MSEC, 2000)
        ret, frame = video.read()
        if not ret:
            st.error("Error reading the video file.")
            return
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        image2 = Image.fromarray(frame)
        annotate_image(image2)

        # Display the uploaded video and the annotated snapshots
        st.video(temp_file_path)
        # st.image(image1, caption="Annotated Snapshot at 0 seconds", use_column_width=True)
        st.image(image2, caption="Annotated Snapshot at 2 seconds", use_column_width=True)

        # Release the video capture
        video.release()
    except Exception as e:
        st.error(f"Error processing video: {e}")

def main():
    st.title("Offline Retailer Sales Targeting App")

    use_webcam = st.checkbox("Use Webcam")
    upload_type = st.radio("Upload Type", ["Image", "Video"])

    if use_webcam:
        if st.button("Capture Frame"):
            try:
                frame = capture_webcam_frame()
                if frame is not None:
                    annotate_image(frame)
                    st.image(frame, caption="Annotated Frame", use_column_width=True)
                else:
                    st.warning("Failed to capture frame from webcam.")
            except Exception as e:
                st.error(f"Error capturing frame: {e}")
    elif upload_type == "Image":
        uploaded_image = st.file_uploader("Upload Image", type=["jpg", "jpeg", "png"])
        if uploaded_image is not None:
            try:
                image = Image.open(uploaded_image)
                st.image(image, caption="Uploaded Image", use_column_width=True)

                if st.button("Annotate"):
                    annotate_image(image)
                    st.image(image, caption="Annotated Image", use_column_width=True)
            except Exception as e:
                st.error(f"Error processing image: {e}")
    elif upload_type == "Video":
        uploaded_video = st.file_uploader("Upload Video", type=["mp4", "mov", "avi"])
        if uploaded_video is not None:
            try:
                annotate_video(uploaded_video)
            except Exception as e:
                st.error(f"Error processing video: {e}")

if __name__ == "__main__":
    main()
