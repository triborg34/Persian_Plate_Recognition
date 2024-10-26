from ultralytics import YOLO
import cv2
import math
import time
import warnings
import torch
import queue
import threading
import logging
import websockets
import asyncio
import base64
from crop_and_licance_saver import crop_and_save_plate

# Configure logging and suppress warnings
logging.getLogger("ultralytics").setLevel(logging.ERROR)
warnings.filterwarnings("ignore", category=FutureWarning)

# Device selection: Check for CUDA, OpenCL, or CPU
if torch.cuda.is_available():
    device = 'cuda'
    print("CUDA is available. Using GPU for inference.")
elif cv2.ocl.haveOpenCL():
    device = 'cpu'
    cv2.ocl.setUseOpenCL(True)
    print("OpenCL is available. Using OpenCL for OpenCV operations.")
else:
    device = 'cpu'
    print("Using CPU for inference.")

# Define class names for characters
classnames = ['car', 'plate']
charclassnames = ['0', '9', 'b', 'd', 'ein', 'ein', 'g', 'gh', 'h', 'n', 's', '1', 'malul', 'n', 's', 'sad', 't', 'ta', 'v', 'y', '2', '3', '4', '5', '6', '7', '8']

# Set the RTSP stream source and initialize YOLO models
source = "rtsp://admin:admin@192.168.1.88:554/mainstream"
model_object = YOLO("weights/best.pt")
model_char = YOLO("weights/yolov8n_char_new.pt")
cap = cv2.VideoCapture(source)
cap.set(cv2.CAP_PROP_BUFFERSIZE, 0)
cap.set(cv2.CAP_PROP_FPS, 30)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1920)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 1080)

# Check if the stream is opened successfully
if not cap.isOpened():
    print("Error: Could not open the RTSP stream.")
    exit()

# Define the output folder for saving cropped plates
output_folder = 'output'

# Initialize frame buffer queue
buffer = queue.Queue(maxsize=10)

# Fill buffer function
def buffer_filler():
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        if buffer.full():
            buffer.get()
        buffer.put(frame)

# Start buffer filler thread
threading.Thread(target=buffer_filler, daemon=True).start()
time.sleep(10)  # Allow buffer to fill for 10 seconds

# WebSocket client to send frames
async def send_frame(ws, img):
    _, buffer = cv2.imencode('.jpg', img)
    jpg_as_text = base64.b64encode(buffer).decode('utf-8')
    await ws.send(jpg_as_text)

async def video_stream():
    async with websockets.connect("ws://localhost:8765") as ws:
        car_id_counter = 1
        while True:
            if buffer.empty():
                await asyncio.sleep(0.1)
                continue

            img = buffer.get()
            output = model_object.predict(img)

            for result in output:
                for box in result.boxes:
                    x1, y1, x2, y2 = map(int, box.xyxy[0].cpu().numpy())
                    cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
                    car_conf = math.ceil((box.conf[0] * 100)) / 100
                    cls_names = int(box.cls[0])
                    print(f"Car ID {car_id_counter} detected with confidence: {car_conf}")

                    if cls_names == 1 and car_conf >= 0.9:
                        plate_img = img[y1:y2, x1:x2]
                        plate_output = model_char.predict(plate_img, conf=0.3)
                        bbox = plate_output[0].boxes.xyxy
                        cls = plate_output[0].boxes.cls
                        keys = cls.cpu().numpy().astype(int)
                        values = bbox[:, 0].cpu().numpy().astype(int)
                        dictionary = list(zip(keys, values))
                        sorted_list = sorted(dictionary, key=lambda x: x[1])
                        char_display = [charclassnames[i[0]] for i in sorted_list]
                        plate_number = ''.join(char_display)

                        if len(plate_number) >= 8:
                            crop_and_save_plate(img, box, plate_number, car_conf, output_folder)
                            cv2.putText(img, f"Car ID: {car_id_counter} | Plate: {plate_number}", (x1, y1 - 10),
                                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
                        car_id_counter += 1

            # Send processed frame to WebSocket
            await send_frame(ws, img)

            # Display frame locally
            cv2.imshow("Detection", img)
            cv2.resizeWindow("Detection", 800, 600)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break

# Start video stream asynchronously
asyncio.run(video_stream())

# Cleanup
cap.release()
cv2.destroyAllWindows()
