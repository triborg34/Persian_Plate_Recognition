from ultralytics import YOLO
import cv2
import math
import time
import datetime
from crop_and_licance_saver import crop_and_save_plate
import warnings
import torch
import queue
import threading
import asyncio
from fastapi import FastAPI, WebSocket
import uvicorn
import logging
# Initialize FastAPI app
app = FastAPI()
logging.getLogger("ultralytics").setLevel(logging.ERROR)
warnings.filterwarnings("ignore", category=FutureWarning)

# Device selection
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

# Set RTSP stream source
source = "rtsp://admin:admin@192.168.1.88:554/substream"  # Replace with your RTSP URL

# Load YOLO models
model_object = YOLO("weights/best.pt")
model_char = YOLO("weights/yolov8n_char_new.pt")

cap = cv2.VideoCapture(source)
cap.set(cv2.CAP_PROP_BUFFERSIZE, 2000)
cap.set(cv2.CAP_PROP_POS_FRAMES, 30)
cap.set(cv2.CAP_PROP_FPS, 5)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1920)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 1080)

# Check if stream is opened
if not cap.isOpened():
    print("Error: Could not open the RTSP stream.")
    exit()

# Initialize frame buffer and car ID counter
buffer = queue.Queue(maxsize=10)
car_id_counter = 1  # Initialize car ID counter
output_folder = 'output'  # Define output folder

# Queue to hold processed frames for WebSocket
frame_queue = queue.Queue(maxsize=10)

# Function to fill the buffer
def buffer_filler():
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        if buffer.full():
            buffer.get()  # Remove oldest frame if buffer is full
        buffer.put(frame)

# Start buffer filler thread
threading.Thread(target=buffer_filler, daemon=True).start()
time.sleep(10)  # Allow buffer to fill

# Main processing function
def process_video():
    global car_id_counter
    charclassnames = ['0', '9', 'b', 'd', 'ein', 'ein', 'g', 'gh', 'h', 'n', 's', '1', 'malul', 'n', 's', 'sad', 't', 'ta', 'v', 'y', '2', '3', '4', '5', '6', '7', '8']

    while True:
        if buffer.empty():
            time.sleep(0.5)
            continue

        img = buffer.get()
        tick = time.time()

        # YOLO inference on the captured frame
        output = model_object.predict(img)

        for result in output:
            for box in result.boxes:
                x1, y1, x2, y2 = map(int, box.xyxy[0].cpu().numpy())
                cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)

                car_conf = math.ceil((box.conf[0] * 100)) / 100
                cls_names = int(box.cls[0])
                print(f"Car ID {car_id_counter} detected with confidence: {car_conf}")

                if cls_names == 1 and car_conf >= 0.9:
                    # Crop plate image and perform character detection
                    plate_img = img[y1:y2, x1:x2]
                    plate_output = model_char.predict(plate_img, conf=0.3)

                    # Extract bounding box, class names, and confidences for characters
                    bbox = plate_output[0].boxes.xyxy
                    cls = plate_output[0].boxes.cls
                    confs_char = plate_output[0].boxes.conf

                    # Print character confidences
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

                    # Use the crop_and_save_plate function to save the car and plate images with car ID
                    if len(plate_number) >= 8 and confidence > 0.6:
                        plate_confidence = box.conf[0].item()  # Assuming plate_conf is the detection confidence of the plate
                        crop_and_save_plate(img, box, plate_number, plate_confidence, output_folder)

                    cv2.putText(img, f"Car ID: {car_id_counter} | Plate: {plate_number}", 
                                (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)

                car_id_counter += 1

        # FPS calculation with zero check for elapsed_time
        tock = time.time()
        elapsed_time = tock - tick
        if elapsed_time > 0:
            fps_text = "FPS: {:.2f}".format(1 / elapsed_time)
            text_size, _ = cv2.getTextSize(fps_text, cv2.FONT_HERSHEY_SIMPLEX, 1, 2)
            fps_text_loc = (img.shape[1] - text_size[0] - 10, text_size[1] + 10)
            cv2.putText(img, fps_text, fps_text_loc, fontFace=cv2.FONT_HERSHEY_SIMPLEX, fontScale=1, 
                        color=(0, 255, 0), thickness=2)

        # Enqueue processed frame for streaming
        if not frame_queue.full():
            frame_queue.put(img)

        # Show detection results (optional)
        cv2.imshow('detection', img)

        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()

# Start processing video in a background thread
threading.Thread(target=process_video, daemon=True).start()

# Endpoint to stream video over WebSocket
@app.websocket("/video")
async def video_stream(websocket: WebSocket):
    await websocket.accept()
    
    try:
        while True:
            if frame_queue.empty():
                await asyncio.sleep(0.01)
                continue

            # Get the next processed frame from the queue
            frame = frame_queue.get()

            # Encode the frame to JPEG format
            _, buffer = cv2.imencode('.jpg', frame)
            
            # Send the frame as binary data over WebSocket
            await websocket.send_bytes(buffer.tobytes())
    
    except Exception as e:
        print(f"Error in WebSocket communication: {e}")
    
    finally:
        await websocket.close()

if __name__ == "__main__":
    uvicorn.run("main_stream:app", host="0.0.0.0", port=8000)
