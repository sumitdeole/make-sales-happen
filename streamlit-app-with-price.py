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
import requests
from streamlit_webrtc import VideoProcessorBase, webrtc_streamer, VideoTransformerBase
import cvzone
import math
from io import BytesIO
from streamlit_webrtc import VideoProcessorBase, webrtc_streamer


# Define the weights folder
weights_folder = "weights/"

# RapidAPI headers
headers = {
    "X-RapidAPI-Key": "4019769c6cmsh0245f46f03d9fecp161a69jsn69acf0a47cdf",
    "X-RapidAPI-Host": "real-time-product-search.p.rapidapi.com"
}

def search_product_price(brand, product_type, country="us", language="en"):
    url = "https://real-time-product-search.p.rapidapi.com/search"
    querystring = {"q": f"{brand} {product_type}", "country": country, "language": language}

    try:
        response = requests.get(url, headers=headers, params=querystring)
        response.raise_for_status()  # Raise an exception for any HTTP error

        # Extract data from the response
        data = response.json().get("data")

        if data:  # Check if data is not empty
            first_product = data[0]

            # Check if typical_price_range is None or not in the expected format
            price_range = first_product.get("typical_price_range")
            if price_range is None or "-" not in price_range:
                # Extract price from offer
                offer = first_product.get("offer")
                if offer:
                    price = offer.get("price")
                    if price:
                        return price
                    else:
                        print("No price found in offer section.")
                else:
                    print("No offer section found.")
            else:
                # Extract typical_price_range and calculate average price
                prices = [float(price.replace('$', '').replace(',', '')) for price in price_range.split("-")]
                average_price = sum(prices) / len(prices)
                return round(average_price, 2)
        else:
            print("No data found in response.")

    except requests.RequestException as e:
        print("Error during request:", e)

    return None

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
            person_count = 0 # Counter for numbering persons
            person_boxes = {} # Dictionary to store bounding boxes for each person
            for score, label, box in zip(detr_results["scores"], detr_results["labels"], detr_results["boxes"]):
                if detr_model.config.id2label[label.item()] == "person":
                    person_count += 1
                    box = [round(i, 2) for i in box.tolist()]
                    person_boxes[f"Person {person_count}"] = box

            # Load the trained YOLO model for product type detection
            product_weight_file_path = "./weights/best_obj_detect_prod_types.pt" # Update with actual path
            product_model = YOLO(product_weight_file_path)

            # Load the trained YOLO model for logo detection
            logo_weight_file_path = "./weights/best_obj_detect_logos.pt" # Update with actual path
            logo_model = YOLO(logo_weight_file_path)

            # Set the detection threshold (e.g., 0.3 for 30% confidence)
            logo_detection_threshold = 0.4

            # Draw bounding boxes and labels for persons on the image
            for person_id, box in person_boxes.items():
                x_min, y_min, x_max, y_max = box
                person_image = image.crop((x_min, y_min, x_max, y_max)) # Crop the image to the person's bounding box

                # Detect product types
                product_results = product_model.predict(person_image)
                product_labels = [product_model.names[int(obj.cls[0])] for obj in product_results[0].boxes]

                # Detect logos
                logo_results = logo_model.predict(person_image, conf=logo_detection_threshold)
                logo_labels = [logo_model.names[int(obj.cls[0])] for obj in logo_results[0].boxes]

                # Construct the label text with price search
                label_text = ""
                unique_combinations = set() # Set to store unique combinations
                for product_label, logo_label in zip(product_labels, logo_labels):
                    combination = (logo_label, product_label)
                    if combination not in unique_combinations:
                        unique_combinations.add(combination)
                        price = search_product_price(logo_label, product_label)
                        if price is not None:
                            label_text += f"{product_label} ({logo_label} of {price})\n"

                if label_text:
                    # Draw the bounding box
                    detr_draw.rectangle(box, outline="blue", width=2)

                    # Calculate label position inside the bounding box (top left corner)
                    font = ImageFont.load_default() # Load default font
                    label_bbox = detr_draw.textbbox((0, 0), label_text, font=font)
                    label_width = label_bbox[2] - label_bbox[0]
                    label_height = label_bbox[3] - label_bbox[1]
                    label_x = x_min + 5 # Offset from the left
                    label_y = y_min + 5 # Offset from the top

                    # Check if the label would be outside the image
                    if label_x + label_width > image.width or label_y + label_height > image.height:
                        # Adjust label position to be outside the image
                        label_x = min(label_x, image.width - label_width)
                        label_y = min(label_y, image.height - label_height)

                    # Draw the label
                    detr_draw.multiline_text((label_x, label_y), label_text, font=font, fill="blue")
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

# Define the WebcamProcessor class
class WebcamProcessor(VideoProcessorBase):
    def __init__(self):
        self.frame_out = None

    def recv(self, img: np.ndarray) -> np.ndarray:
        # Detect product types using the YOLO model
        product_results = product_model.predict(img)
        product_labels = [product_model.names[int(obj.cls[0])] for obj in product_results[0].boxes]

        # Detect logos using the YOLO model
        logo_results = logo_model.predict(img, conf=0.4)
        logo_labels = [logo_model.names[int(obj.cls[0])] for obj in logo_results[0].boxes]

        # Annotate the webcam feed with detected product types and logos
        for label in product_labels:
            bbox = get_bbox_for_label(label)
            x1, y1, x2, y2 = bbox
            cv2.rectangle(img, (x1, y1), (x2, y2), (255, 0, 255), 3)
            cvzone.putTextRect(img, label, (max(0, x1), max(35, y1)), scale=1, thickness=1)

        for label in logo_labels:
            bbox = get_bbox_for_label(label)
            x1, y1, x2, y2 = bbox
            cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 3)
            cvzone.putTextRect(img, label, (max(0, x1), max(35, y1)), scale=1, thickness=1)

        return img
    
def get_bbox_for_label(label):
    # Implement your logic to get the bounding box for the given label
    # This could involve using the YOLO model predictions or any other object detection method
    # For demonstration, let's assume the bounding box is hardcoded
    x1 = 50
    y1 = 50
    x2 = 150
    y2 = 150
    return x1, y1, x2, y2

def main():
    st.title("Make Sales Happen: Offline Retailer Sales Targeting App")

    # Use columns to create a left and right layout
    left_col, right_col = st.columns([3, 3])

    with left_col:
        # Browse and Selection
        upload_type = st.radio("Upload Type", ["Webcam", "Image", "Video"])

        if upload_type == "Webcam":
            # Show the webcam feed with product type and logo detection
            webrtc_streamer(key="webcam", video_processor_factory=WebcamProcessor)

        elif upload_type == "Image":
            uploaded_image = st.file_uploader("Upload Image", type=["jpg", "jpeg", "png"])
            if uploaded_image is not None:
                image = Image.open(uploaded_image)
                st.image(image, caption="Uploaded Image", use_column_width=True)
                if st.button("Annotate"):
                    # Annotate the image
                    annotate_image(image)
                    # Display the annotated image in the right column
                    with right_col:
                        st.image(image, caption="Annotated Image", use_column_width=True)

        elif upload_type == "Video":
            uploaded_video = st.file_uploader("Upload Video", type=["mp4", "mov", "avi"])
            if uploaded_video is not None:
                # Process the uploaded video
                annotate_video(uploaded_video)
                
if __name__ == "__main__":
    main()
