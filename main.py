from ultralytics import YOLO
import cv2
import math
import time
import datetime
from crop_and_licance_saver import crop_and_save_plate
import warnings
import torch

warnings.filterwarnings("ignore", category=FutureWarning)

# Device selection: Check for CUDA, OpenCL, or CPU
if torch.cuda.is_available():
    device = 'cuda'
    print("CUDA is available. Using GPU for inference.")
elif cv2.ocl.haveOpenCL():
    device = 'cpu'  # PyTorch inference will use CPU, OpenCV uses OpenCL
    cv2.ocl.setUseOpenCL(True)
    print("OpenCL is available. Using OpenCL for OpenCV operations, but inference will be on CPU.")
else:
    device = 'cpu'
    print("Neither CUDA nor OpenCL is available. Using CPU for inference.")

# Get the current timestamp for output names
timestamp = datetime.datetime.now().strftime("%Y%m%d%H%M%S")
classnames = ['car', 'plate']
charclassnames = ['0', '9', 'b', 'd', 'ein', 'ein', 'g', 'gh', 'h', 'n', 's', '1', 'malul', 'n', 's', 'sad', 't', 'ta', 'v', 'y', '2', '3', '4', '5', '6', '7', '8']

# Set the RTSP stream source
source = "rtsp://admin:admin@192.168.1.88:554/substream"  # Replace with your RTSP URL

# Load YOLOv8 models for object and character detection
model_object = YOLO("weights/best.pt")  # Model is automatically loaded to the right device
model_char = YOLO("weights/yolov8n_char_new.pt")  # Model is automatically loaded to the right device

cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_BUFFERSIZE, 2000)
cap.set(cv2.CAP_PROP_POS_FRAMES, 30)

# Check if the stream is opened successfully
if not cap.isOpened():
    print("Error: Could not open the RTSP stream.")
    exit()

# Define the output folder for saving cropped plates
output_folder = 'output'

# Retry parameters
retry_delay = 5  # seconds to wait before retrying
max_retries = 5  # maximum number of retries
retry_count = 0  # retry counter

while retry_count < max_retries:
    success, img = cap.read()
    if success:
        tick = time.time()

        # Perform YOLO inference on the captured frame (img)
        output = model_object.predict(img)  # Run inference

        for result in output:
            for box in result.boxes:
                x1, y1, x2, y2 = map(int, box.xyxy[0].cpu().numpy())  # Convert to integers
                cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)

                # Print car confidence
                car_conf = math.ceil((box.conf[0] * 100)) / 100
                cls_names = int(box.cls[0])
                print(f"Car detected with confidence: {car_conf}")

                if cls_names == 1 and car_conf >= 0.9:  # Only proceed when confidence is above 0.9
                    # Detect characters with the YOLO model for plates
                    plate_img = img[y1:y2, x1:x2]  # Crop the plate image
                    plate_output = model_char.predict(plate_img, conf=0.3)  # Perform character detection

                    # Extract bounding box, class names, and confidences for characters
                    bbox = plate_output[0].boxes.xyxy
                    cls = plate_output[0].boxes.cls
                    confs_char = plate_output[0].boxes.conf  # Extract character confidences
                    
                    # Print the confidence for each detected character
                    print("Character confidences:")
                    for confidence in confs_char:
                        print(f"Character detected with confidence: {confidence:.2f}")

                    # Sort characters left to right
                    keys = cls.cpu().numpy().astype(int)
                    values = bbox[:, 0].cpu().numpy().astype(int)
                    dictionary = list(zip(keys, values))
                    sorted_list = sorted(dictionary, key=lambda x: x[1])

                    # Convert characters to string
                    char_display = []
                    for i in sorted_list:
                        char_class = i[0]
                        char_display.append(charclassnames[char_class])

                    plate_number = ''.join(char_display)

                    # Use the crop_and_save_plate function to save the car and plate images
                    if len(plate_number) >= 8:
                        crop_and_save_plate(img, box, plate_number,confidence, output_folder)

                    # Display the detected plate characters on the image
                    if len(char_display) >= 8:
                        cv2.line(img, (max(40, x1 - 25), max(40, y1 - 10)), (x2 + 25, y1 - 10), (0, 0, 0), 20, lineType=cv2.LINE_AA)
                        cv2.putText(img, plate_number, (max(40, x1 - 15), max(40, y1 - 5)), fontFace=cv2.FONT_HERSHEY_TRIPLEX, fontScale=0.5, color=(10, 50, 255), thickness=1, lineType=cv2.LINE_AA)

        tock = time.time()
        elapsed_time = tock - tick
        fps_text = "FPS: {:.2f}".format(1 / elapsed_time)
        text_size, _ = cv2.getTextSize(fps_text, cv2.FONT_HERSHEY_SIMPLEX, 1, 2)
        fps_text_loc = (img.shape[1] - text_size[0] - 10, text_size[1] + 10)
        cv2.putText(img, fps_text, fps_text_loc, fontFace=cv2.FONT_HERSHEY_TRIPLEX, fontScale=1, color=(10, 50, 255), thickness=2, lineType=cv2.LINE_AA)

        # Show detection
        cv2.imshow('detection', img)

        # Break loop on 'q' key press
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

        # Reset retry count after successful read
        retry_count = 0
    else:
        retry_count += 1
        print(f"Error: Could not get any frames from the RTSP stream. Retrying in {retry_delay} seconds... (Attempt {retry_count} of {max_retries})")
        time.sleep(retry_delay)

        if retry_count >= max_retries:
            print("Max retries reached. Exiting...")
            break

cap.release()
cv2.destroyAllWindows()
